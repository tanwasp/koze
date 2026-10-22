-- Koze data collection: run once in the Supabase SQL editor (safe to re-run).
--
-- Anyone can use the transcriber; sharing clips needs an account (passwordless email
-- code). The page holds only the project's public key. With these rules a signed-in
-- contributor can add clips under their own id, read and delete their own clips, and
-- edit a few fields of their profile. They can't see anyone else's data or give
-- themselves a role. Admins can read everything. The service-role key, used only by
-- tools/export_dataset.py on your own machine, bypasses all of this.

-- ------------------------------------------------------------------ profiles

create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  created_at timestamptz not null default now(),
  display_name text check (char_length(display_name) <= 60),
  -- contributor: anyone who signs up. collector: a volunteer or paid transcriber you
  -- approved. admin: can read every clip. Only you change this, in the dashboard.
  role text not null default 'contributor' check (role in ('contributor', 'collector', 'admin')),
  consent_version text check (char_length(consent_version) <= 20),
  consent_at timestamptz,
  open_release boolean not null default false,   -- may their clips go in a public dataset?
  speaker jsonb check (speaker is null or (jsonb_typeof(speaker) = 'object'
    and speaker - array['age', 'gender', 'region', 'first_language'] = '{}'::jsonb
    and pg_column_size(speaker) <= 1000))
);

create or replace function public.is_admin() returns boolean
language sql stable security definer set search_path = '' as $$
  select exists (select 1 from public.profiles where id = auth.uid() and role = 'admin')
$$;

alter table public.profiles enable row level security;
revoke all on public.profiles from anon, authenticated;
grant select on public.profiles to authenticated;
grant update (display_name) on public.profiles to authenticated;
drop policy if exists "read own profile" on public.profiles;
create policy "read own profile" on public.profiles for select to authenticated
  using (id = auth.uid() or public.is_admin());
drop policy if exists "edit own profile" on public.profiles;
create policy "edit own profile" on public.profiles for update to authenticated
  using (id = auth.uid()) with check (id = auth.uid());

create or replace function public.handle_new_user() returns trigger
language plpgsql security definer set search_path = '' as $$
begin
  insert into public.profiles (id) values (new.id) on conflict do nothing;
  return new;
end $$;
drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created after insert on auth.users
  for each row execute function public.handle_new_user();

-- Consent is recorded here, with the time, not just in the browser.
create or replace function public.give_consent(version text, open boolean, details jsonb) returns void
language sql security definer set search_path = '' as $$
  update public.profiles
  set consent_version = version, consent_at = now(), open_release = open,
      speaker = nullif(details, '{}'::jsonb)
  where id = auth.uid()
$$;
create or replace function public.withdraw_consent() returns void
language sql security definer set search_path = '' as $$
  update public.profiles set consent_version = null, consent_at = null, open_release = false
  where id = auth.uid()
$$;
-- Deletes the account; the profile and clip rows go with it (on delete cascade). The page
-- deletes the audio files first; export_dataset.py --clean removes any it missed.
create or replace function public.delete_my_account() returns void
language sql security definer set search_path = '' as $$
  delete from auth.users where id = auth.uid()
$$;
revoke all on function public.give_consent(text, boolean, jsonb), public.withdraw_consent(),
  public.delete_my_account(), public.is_admin() from public, anon;
grant execute on function public.give_consent(text, boolean, jsonb), public.withdraw_consent(),
  public.delete_my_account(), public.is_admin() to authenticated;

-- ------------------------------------------------------------------ clips

create table if not exists public.clips (
  id uuid primary key,
  created_at timestamptz not null default now(),
  user_id uuid not null default auth.uid() references public.profiles (id) on delete cascade,
  audio_path text not null,                      -- <user_id>/<id>.wav in the "clips" bucket: 16 kHz mono 16-bit
  source text not null check (source in ('recording', 'upload')),
  duration_s real not null check (duration_s > 0 and duration_s <= 300),
  model_id text not null check (char_length(model_id) <= 40),
  model_card text not null check (char_length(model_card) <= 80),
  backend text check (backend in ('webgpu', 'wasm')),
  model_text text not null check (char_length(model_text) <= 10000),   -- the model's output, untouched
  text text not null check (char_length(text) between 1 and 10000),    -- the contributor's version, exactly as typed
  spelling text check (spelling in ('larmoni', 'free')),                -- did they aim for Grafi Larmoni spelling?
  languages text[] check (languages <@ array['kreol', 'french', 'english', 'bhojpuri', 'other']),
  rating smallint check (rating between 1 and 3),                      -- 3 good, 2 some mistakes, 1 mostly wrong
  issues text[] check (issues <@ array['words', 'spelling', 'names', 'numbers', 'mixed', 'noise', 'missing', 'extra']),
  comment text check (char_length(comment) <= 2000),
  own_voice boolean not null,                    -- is the contributor one of the speakers?
  permission_confirmed boolean not null check (permission_confirmed),  -- everyone audible agreed
  -- Copied from the profile when the clip is added, so each clip keeps the terms it was given under.
  speaker jsonb,
  consent_version text not null,
  open_release boolean not null,
  constraint audio_path_matches check (audio_path = user_id::text || '/' || id::text || '.wav')
);

create or replace function public.stamp_clip() returns trigger
language plpgsql security definer set search_path = '' as $$
declare p public.profiles;
begin
  select * into p from public.profiles where id = new.user_id;
  if p.consent_at is null then
    raise exception 'Agree to the consent terms before sharing' using errcode = '42501';
  end if;
  new.created_at := now();
  new.consent_version := p.consent_version;
  new.open_release := p.open_release;
  new.speaker := case when new.own_voice then p.speaker end;
  return new;
end $$;
drop trigger if exists stamp_clip on public.clips;
create trigger stamp_clip before insert on public.clips for each row execute function public.stamp_clip();

alter table public.clips enable row level security;
revoke all on public.clips from anon, authenticated;
grant insert (id, audio_path, source, duration_s, model_id, model_card, backend, model_text, text, spelling,
  languages, rating, issues, comment, own_voice, permission_confirmed) on public.clips to authenticated;
grant select, delete on public.clips to authenticated;
drop policy if exists "add own clips" on public.clips;
create policy "add own clips" on public.clips for insert to authenticated with check (user_id = auth.uid());
drop policy if exists "read own clips" on public.clips;
create policy "read own clips" on public.clips for select to authenticated
  using (user_id = auth.uid() or public.is_admin());
drop policy if exists "delete own clips" on public.clips;
create policy "delete own clips" on public.clips for delete to authenticated using (user_id = auth.uid());

-- ------------------------------------------------------------------ audio

-- Private, WAV only, 10 MB (about 5 minutes) per file.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('clips', 'clips', false, 10485760, array['audio/wav'])
on conflict (id) do update set public = false, file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

-- Each contributor writes only to <their id>/<uuid>.wav, and reads and deletes only their
-- own folder. No update policy, so nothing can be overwritten.
drop policy if exists "upload own audio" on storage.objects;
create policy "upload own audio" on storage.objects for insert to authenticated
with check (bucket_id = 'clips' and name ~ ('^' || auth.uid()::text
  || '/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.wav$'));
drop policy if exists "read own audio" on storage.objects;
create policy "read own audio" on storage.objects for select to authenticated
using (bucket_id = 'clips' and (split_part(name, '/', 1) = auth.uid()::text or public.is_admin()));
drop policy if exists "delete own audio" on storage.objects;
create policy "delete own audio" on storage.objects for delete to authenticated
using (bucket_id = 'clips' and split_part(name, '/', 1) = auth.uid()::text);
