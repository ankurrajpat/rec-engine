-- Example queries over artifacts/jobmatch/jobmatch.db (SQLite).  sqlite3 artifacts/jobmatch/jobmatch.db < scripts/jobmatch_sql.sql

-- 1. Label mix per split (jobs.split joins to fit_labels)
SELECT j.split, f.grade, COUNT(*) AS pairs, ROUND(100.0*COUNT(*)/SUM(COUNT(*)) OVER (PARTITION BY j.split),1) AS pct
FROM fit_labels f JOIN jobs j USING(jid) GROUP BY j.split, f.grade ORDER BY j.split, f.grade;

-- 2. Jobs with the highest share of Good Fit candidates (min 10 labelled candidates)
SELECT jid, COUNT(*) AS n, ROUND(AVG(grade=2),2) AS good_share
FROM fit_labels GROUP BY jid HAVING n>=10 ORDER BY good_share DESC LIMIT 5;

-- 3. Rank candidates inside each job by grade with a window function, keep the top 3
SELECT * FROM (
  SELECT jid, rid, grade, ROW_NUMBER() OVER (PARTITION BY jid ORDER BY grade DESC, rid) AS rn FROM fit_labels
) WHERE rn<=3 LIMIT 9;

-- 4. Candidates labelled against many jobs but never a Good Fit (hard-to-place profiles)
SELECT rid, COUNT(*) AS jobs_seen FROM fit_labels GROUP BY rid HAVING MAX(grade)<2 AND jobs_seen>=8 ORDER BY jobs_seen DESC LIMIT 5;
