-- Koze data collection: run once in the Supabase SQL editor (safe to re-run).
--
-- The web page holds only the project's public (anon/publishable) key. With these rules
-- that key can add a clip and upload its audio, and nothing else: it can't list, read,
-- change or overwrite anything. A contributor can withdraw their own clip with the
-- random token their browser kept when they shared it. Reading the data needs the
-- service-role key, which only tools/export_dataset.py uses, on your own machine.

create table if not exists public.clips (
  id uuid primary key,
  created_at timestamptz not null default now(),
  audio_path text not null,                      -- <id>.wav in the "clips" bucket: 16 kHz mono 16-bit
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
  speaker jsonb check (speaker is null or (jsonb_typeof(speaker) = 'object'
    and speaker - array['age', 'gender', 'region', 'first_language'] = '{}'::jsonb
    and pg_column_size(speaker) <= 1000)),
  permission_confirmed boolean not null check (permission_confirmed),  -- everyone audible agreed
  consent_version text not null check (char_length(consent_version) <= 20),
  open_release boolean not null default false,                         -- may go in a public dataset
  delete_token_hash text not null check (delete_token_hash ~ '^[0-9a-f]{64}$'),
  constraint audio_path_matches_id check (audio_path = id::text || '.wav')
);

alter table public.clips enable row level security;
revoke all on public.clips from anon, authenticated;
-- Insert only, and not created_at, so the server sets it.
grant insert (id, audio_path, source, duration_s, model_id, model_card, backend, model_text, text, spelling,
  languages, rating, issues, comment, speaker, permission_confirmed, consent_version, open_release,
  delete_token_hash) on public.clips to anon, authenticated;
drop policy if exists "anyone can add a clip" on public.clips;
create policy "anyone can add a clip" on public.clips for insert to anon, authenticated with check (true);

-- Withdraw: delete the row if the token matches. The audio is then an orphan, and
-- tools/export_dataset.py --clean deletes orphaned audio.
create or replace function public.withdraw_clip(clip_id uuid, token text) returns boolean
language sql security definer set search_path = '' as $$
  with gone as (
    delete from public.clips
    where id = clip_id and delete_token_hash = encode(sha256(convert_to(token, 'UTF8')), 'hex')
    returning 1)
  select exists (select 1 from gone)
$$;
revoke all on function public.withdraw_clip(uuid, text) from public;
grant execute on function public.withdraw_clip(uuid, text) to anon, authenticated;

-- Audio bucket: private, WAV only, 10 MB (about 5 minutes) per file.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('clips', 'clips', false, 10485760, array['audio/wav'])
on conflict (id) do update set public = false, file_size_limit = excluded.file_size_limit,
  allowed_mime_types = excluded.allowed_mime_types;

-- Upload only, to <uuid>.wav. No select or update policy, so nobody can list, read or
-- overwrite files with the public key.
drop policy if exists "anyone can upload a clip" on storage.objects;
create policy "anyone can upload a clip" on storage.objects for insert to anon, authenticated
with check (bucket_id = 'clips'
  and name ~ '^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.wav$');
