-- NeuroMap · Global Brain — Supabase schema (run once in SQL Editor)
-- Anonymous, insert-only. Raw rows are never readable by the public;
-- the site only calls gb_stats(), which returns aggregates and hides
-- any country with fewer than 5 participants.

create table if not exists public.gb_results (
  id         bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  country    text     not null check (country ~ '^[A-Z]{2}$'),
  age_band   smallint check (age_band between 1 and 7),
  hour       smallint check (hour between 0 and 23),
  rt         real     check (rt between 100 and 1500),
  stroop     real     check (stroop between -300 and 1000),
  subit      smallint check (subit between 1 and 8),
  drm        boolean,
  lang       text     check (char_length(lang) <= 5)
);
create index if not exists gb_results_country on public.gb_results(country);

alter table public.gb_results enable row level security;
drop policy if exists gb_insert on public.gb_results;
create policy gb_insert on public.gb_results for insert to anon with check (true);
-- no SELECT policy → nobody can read raw rows through the public key

create or replace function public.gb_stats() returns json
language sql stable security definer set search_path = public as $$
  select json_build_object(
    'total', (select count(*) from gb_results),
    'world', (select json_build_object(
        'rt',     round(percentile_cont(.5) within group (order by rt)::numeric),
        'stroop', round(percentile_cont(.5) within group (order by stroop)::numeric),
        'subit',  round(avg(subit)::numeric, 1),
        'drm',    round(avg(case when drm then 1.0 when drm is not null then 0.0 end)::numeric, 2))
      from gb_results),
    'countries', (select coalesce(json_agg(c), '[]'::json) from (
        select country, count(*) as n,
          round(percentile_cont(.5) within group (order by rt)::numeric)     as rt,
          round(percentile_cont(.5) within group (order by stroop)::numeric) as stroop,
          round(avg(subit)::numeric, 1) as subit,
          round(avg(case when drm then 1.0 when drm is not null then 0.0 end)::numeric, 2) as drm
        from gb_results group by country having count(*) >= 5) c),
    'hours', (select coalesce(json_agg(h), '[]'::json) from (
        select hour, count(*) as n, round(percentile_cont(.5) within group (order by rt)::numeric) as rt
        from gb_results where rt is not null and hour is not null group by hour having count(*) >= 5) h)
  );
$$;
revoke all on function public.gb_stats() from public;
grant execute on function public.gb_stats() to anon;
