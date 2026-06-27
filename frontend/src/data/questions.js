// Questions data with correct SQL queries
// IMPORTANT: Do NOT add semicolons (;) at the end of queries - the Calcite SQL parser doesn't support them

// ========== BEERS SCHEMA ==========
export const beersQuestions = [
  // { old one!!!
  //   id: 'q1',
  //   label: 'Q1',
  //   question: "For each beer 𝑏 that Amy likes and each bar 𝑟 frequented by Amy that serves 𝑏, show the rank of 𝑟 among all bars serving 𝑏 according to price ",
  //   // User input: SELECT s2.beer, s2.bar, COUNT(*) FROM Likes, Serves s1, Serves s2 WHERE drinker = 'Amy' AND Likes.beer = s1.beer AND Likes.beer = s2.beer AND s1.price > s2.price GROUP BY s2.beer, s2.bar
  //   correctQuery: "SELECT L.beer, S1.bar, COUNT(*) FROM Likes L, Frequents F, Serves S1, Serves S2 WHERE L.drinker = F.drinker AND F.bar = S1.bar AND L.beer = S1.beer AND S1.beer = S2.beer AND S1.price <= S2.price GROUP BY F.drinker, L.beer, S1.bar HAVING F.drinker = 'Amy'"
  // },
  
  // ------------new one------------
  {
    id: 'q1',
    label: 'Q1',
    question: "For each beer 𝑏 that Amy likes and each bar 𝑟 frequented by Amy that serves 𝑏, show the rank of 𝑟 among all bars serving 𝑏 according to price ",
    // User input: SELECT s2.beer, s2.bar, COUNT(*) FROM Likes, Frequents, Serves s1, Serves s2 WHERE drinker = 'Amy' AND Likes.beer = s1.beer AND Likes.beer = s2.beer AND s1.price > s2.price AND Frequents.bar = s2.bar AND Frequents.drinker = Likes.drinker GROUP BY s2.beer, s2.bar
    correctQuery: "SELECT L.beer, S1.bar, COUNT(*) FROM Likes L, Frequents F1, Frequents F2, Serves S1, Serves S2 WHERE L.drinker = F1.drinker AND F1.bar = S1.bar AND L.beer = S1.beer AND S1.beer = S2.beer AND S1.price <= S2.price AND F2.drinker = L.drinker AND F2.bar = S2.bar GROUP BY F1.drinker, L.beer, S1.bar HAVING F1.drinker = 'Amy'"
  },
  // {
  //   id: 'q2',
  //   label: 'Q2',
  //   question: 'Find names and addresses of bars that serve "Budweiser" and have a price greater than $2.20.',
  //   correctQuery: "SELECT name, address FROM Bar, Serves WHERE Bar.name = Serves.bar AND beer = 'Budweiser' AND price > 2.20"
  // },
  {
    id: 'q2',
    label: 'Q2',
    question: 'Find the names of bars that serve at least two distinct beers liked by Eve. ',
    // User input: SELECT s.bar FROM Serves s, Likes l WHERE l.beer = s.beer AND l.drinker = 'Eve'GROUP BY s.bar HAVING COUNT(*) >= 2
    correctQuery: "SELECT s.bar FROM Serves s WHERE EXISTS(SELECT * FROM Likes l WHERE l.beer = s.beer AND l.drinker = 'Eve') GROUP BY s.bar HAVING COUNT(*) >= 2"
  },
  {
    id: 'q3',
    label: 'Q3',
    question: 'Which drinkers like the beer Co  rona and go to the bar James Joyce Pub at least twice a week?',
    correctQuery: "SELECT Likes.drinker FROM Likes, Frequents WHERE Likes.beer = 'Corona' AND Likes.drinker = Frequents.drinker AND Frequents.bar = 'James Joyce Pub' AND Frequents.times_a_week >= 2"
  },
  {
    id: 'q4',
    label: 'Q4',
    question: 'Which drinkers like at least two different beers?',
    correctQuery: "SELECT distinct drinker FROM Likes GROUP BY drinker HAVING COUNT(beer) >= 2"
  },
  {
    id: 'q5',
    label: 'Q5',
    question: 'Which drinkers have liked Budweiser at least once?',
    correctQuery: "SELECT drinker FROM Likes WHERE beer = 'Budweiser' GROUP BY drinker HAVING COUNT(*) >= 1"
  },
  {
    id: 'q6',
    label: 'Q6',
    question: 'Find all drinkers like at least one beer?',
    correctQuery: "SELECT drinker FROM Likes"
  },
  {
    id: 'q7',
    label: 'Q7',
    question: 'Corret answer: SELECT drinker FROM Frequents GROUP BY drinker HAVING SUM(times_a_week + times_a_week) >= 3',
    correctQuery: "SELECT drinker FROM Frequents GROUP BY drinker HAVING SUM(times_a_week + times_a_week) >= 3"
  },
  {
    id: 'q8',
    label: 'Q8',
    question: 'Which drinkers like a beer that is served at a bar, and who also frequent (visit) that bar, such that either: the beer is served for less than $5, OR the drinker goes to the bar more than 3 times per week AND the beer’s price at that bar is between $5 and $7.',
    correctQuery: "SELECT L.drinker FROM Likes L, Serves S, Frequents F WHERE L.beer = S.beer AND L.drinker = F.drinker AND (S.price < 5.0 OR (F.times_a_week > 3 AND S.price > 5 AND S.price < 7))"
  },
  {
    id: 'q9',
    label: 'Q9',
    question: "List all drinkers who appear in the Likes table, along with the bars they frequent and how many times per week they visit each bar.", // SELECT * FROM Frequents WHERE EXISTS (SELECT * FROM Likes WHERE Likes.drinker = Frequents.drinker)
    correctQuery: "SELECT distinct Frequents.drinker, Frequents.bar, Frequents.times_a_week FROM Frequents, Likes WHERE Likes.drinker = Frequents.drinker"
  },
  {
    id: 'q10',
    label: 'Q10',
    question: "If Bob appears in the Frequents table, list all distinct (drinker, bar, times_a_week) records from Frequents", // SELECT * FROM Frequents WHERE EXISTS (SELECT * FROM Frequents WHERE Frequents.drinker = 'Bob')
    correctQuery: "SELECT DISTINCT Frequents.drinker, Frequents.bar, Frequents.times_a_week FROM Frequents, Frequents AS Frequents_1 WHERE Frequents_1.drinker = 'Bob'"
  },
  {
    id: 'q11',
    label: 'Q11',
    question: "Compute three times the total number of weekly bar visits over all frequenting records whose drinkers appear in the Drinker table.", // SELECT sum(f.times_a_week * 2) + sum(f.times_a_week) FROM Drinker d, Frequents f WHERE d.name = f.drinker
    correctQuery: "SELECT sum(f.times_a_week) * 3 FROM Drinker d, Frequents f WHERE d.name = f.drinker"
  },
  {
    id: 'q12',
    label: 'Q12',
    question: "List all distinct combinations of drinker information and frequenting information for drinkers who appear in the Frequents table.", 
    correctQuery: "SELECT Distinct * FROM Drinker d, Frequents f WHERE d.name = f.drinker"
  },
  {
    id: 'q13',
    label: 'Q13',
    question: "", 
    correctQuery: "SELECT DISTINCT f1.drinker, f1.bar, f1.times_a_week FROM Frequents AS f1, Frequents AS f2 WHERE f2.drinker = f1.drinker"
  },
  {
    id: 'q14',
    label: 'Q14',
    question: "", //SELECT * FROM Frequents WHERE times_a_week < SOME (SELECT times_a_week FROM Frequents WHERE bar = 'Pub')
    correctQuery: "SELECT DISTINCT Frequents.drinker, Frequents.bar, Frequents.times_a_week FROM Frequents, Frequents AS Frequents_1 WHERE (Frequents.times_a_week < Frequents_1.times_a_week AND Frequents_1.bar = 'Pub')"
  },
  {
    id: 'q15',
    label: 'Q15',
    question: "", 
    correctQuery: ""
  },
  {
    id: 't1',
    label: ' Test1',
    question: 'Which drinkers like a beer that is served at a bar, and who also frequent that bar, such that either the beer is served for less than $5, or they visit that bar more than 3 times per week and the beer is priced between $5 and $7?',
    correctQuery: "SELECT L.drinker FROM Likes L, Serves S, Frequents F WHERE L.beer = S.beer AND L.drinker = F.drinker AND (S.price < 5.0 OR (F.times_a_week > 3 AND S.price > 5 AND S.price < 7))"
  },
  {
    id: 't2',
    label: ' Test2',
    question: 'Which bars serve a beer that is liked by drinkers who both frequent the bar more than twice a week and like at least one beer served for more than $6 at that same bar?',
    correctQuery: "SELECT S.bar FROM Serves S, Likes L, Frequents F WHERE L.beer = S.beer AND L.drinker = F.drinker AND F.bar = S.bar AND F.times_a_week > 2 AND S.price > 6"
  },
  {
    id: 't3',
    label: ' Test3',
    question: 'Which drinkers like beers brewed by more than one distinct brewer, considering only beers that are served at bars they frequent at least 3 times per week?',
    correctQuery: "SELECT L.drinker FROM Likes L, Beer B, Serves S, Frequents F WHERE L.beer = B.name AND B.name = S.beer AND L.drinker = F.drinker AND S.bar = F.bar AND F.times_a_week >= 3 GROUP BY L.drinker HAVING COUNT(DISTINCT B.brewer) > 1"
  },
  {
    id: 't4',
    label: ' Test4',
    question: 'Which bars serve beers that are liked by drinkers who visit the bar fewer than 3 times per week but who like at least two different beers served there?',
    correctQuery: "SELECT F.bar FROM Frequents F, Likes L, Serves S WHERE F.drinker = L.drinker AND L.beer = S.beer AND S.bar = F.bar AND F.times_a_week < 3 GROUP BY F.bar, F.drinker HAVING COUNT(DISTINCT L.beer) >= 2"
  },
  {
    id: 't5',
    label: ' Test5',
    question: 'Which drinkers frequent at least one bar that serves both a cheap beer under $4 and an expensive beer above $8, regardless of whether they like those beers?',
    correctQuery: "SELECT F.drinker FROM Frequents F, Serves S1, Serves S2 WHERE F.bar = S1.bar AND F.bar = S2.bar AND S1.price < 4 AND S2.price > 8"
  },
  {
    id: 't6',
    label: ' Test6',
    question: 'Which beers are liked by drinkers who only frequent bars that serve the beer for less than $5 on average?',
    correctQuery: "SELECT L.beer FROM Likes L, Serves S, Frequents F WHERE L.drinker = F.drinker AND F.bar = S.bar AND L.beer = S.beer GROUP BY L.beer, L.drinker HAVING AVG(S.price) < 5"
  },
  {
    id: 't7',
    label: ' Test7',
    question: 'Which bars serve beers liked by drinkers who like at least three distinct beers, all brewed by different brewers?',
    correctQuery: "SELECT S.bar FROM Serves S, Likes L, Beer B WHERE L.beer = B.name AND S.beer = L.beer GROUP BY S.bar, L.drinker HAVING COUNT(DISTINCT B.brewer) >= 3"
  },
  {
    id: 't8',
    label: ' Test8',
    question: 'Which drinkers like a beer that is served at at least two different bars, where at least one bar serves it for less than $4 and another bar serves it for more than $7?',
    correctQuery: "SELECT L.drinker FROM Likes L, Serves S1, Serves S2 WHERE L.beer = S1.beer AND L.beer = S2.beer AND S1.bar <> S2.bar AND S1.price < 4 AND S2.price > 7"
  },
  {
    id: 't9',
    label: ' Test9',
    question: 'Which beers are served at bars that are frequented by more than three distinct drinkers who like that beer?',
    correctQuery: "SELECT S.beer FROM Serves S, Frequents F, Likes L WHERE S.beer = L.beer AND L.drinker = F.drinker AND F.bar = S.bar GROUP BY S.beer HAVING COUNT(DISTINCT F.drinker) > 3"
  },
  {
    id: 't10',
    label: ' Test10',
    question: 'Which drinkers only like beers brewed by a single brewer but frequent at least two bars that serve those beers?',
    correctQuery: "SELECT L.drinker FROM Likes L, Beer B, Serves S, Frequents F WHERE L.beer = B.name AND L.beer = S.beer AND L.drinker = F.drinker AND S.bar = F.bar GROUP BY L.drinker HAVING COUNT(DISTINCT B.brewer) = 1 AND COUNT(DISTINCT F.bar) >= 2"
  },

  // ========== Subquery Test Cases ==========
  // Case 1: Correct has EXISTS/SOME, User uses JOIN
  {
    id: 'sub1',
    label: 'Sub1',
    question: "List all drinkers who appear in the Likes table, along with the bars they frequent and how many times per week they visit each bar.",
    // User input: SELECT DISTINCT Frequents.drinker, Frequents.bar, Frequents.times_a_week FROM Frequents, Likes WHERE Likes.drinker = Frequents.drinker
    correctQuery: "SELECT * FROM Frequents WHERE EXISTS (SELECT * FROM Likes WHERE Likes.drinker = Frequents.drinker)"
  },
  {
    id: 'sub2',
    label: 'Sub2',
    question: "If Bob appears in the Frequents table, list all distinct (drinker, bar, times_a_week) records from Frequents.",
    // User input: SELECT DISTINCT Frequents.drinker, Frequents.bar, Frequents.times_a_week FROM Frequents, Frequents AS Frequents_1 WHERE Frequents_1.drinker = 'Bob'
    correctQuery: "SELECT * FROM Frequents WHERE EXISTS (SELECT * FROM Frequents AS f2 WHERE f2.drinker = 'Bob')"
  },
  {
    id: 'sub3',
    label: 'Sub3',
    question: "Find all frequenting records where the visit frequency is less than some visit frequency at the bar 'Pub'.",
    // User input: SELECT DISTINCT Frequents.drinker, Frequents.bar, Frequents.times_a_week FROM Frequents, Frequents AS Frequents_1 WHERE Frequents.times_a_week < Frequents_1.times_a_week AND Frequents_1.bar = 'Pub'
    correctQuery: "SELECT * FROM Frequents WHERE times_a_week < SOME (SELECT times_a_week FROM Frequents WHERE bar = 'Pub')"
  },

  // Case 2: Correct uses JOIN, User uses EXISTS/SOME
  {
    id: 'sub4',
    label: 'Sub4',
    question: "List all drinkers who appear in the Likes table, along with the bars they frequent.",
    // User input: SELECT * FROM Frequents WHERE EXISTS (SELECT * FROM Likes WHERE Likes.drinker = Frequents.drinker)
    correctQuery: "SELECT DISTINCT Frequents.drinker, Frequents.bar, Frequents.times_a_week FROM Frequents, Likes WHERE Likes.drinker = Frequents.drinker"
  },
  {
    id: 'sub5',
    label: 'Sub5',
    question: "Find frequenting records where some drinker named 'Bob' exists in Frequents.",
    // User input: SELECT * FROM Frequents WHERE EXISTS (SELECT * FROM Frequents AS f2 WHERE f2.drinker = 'Bob')
    correctQuery: "SELECT DISTINCT Frequents.drinker, Frequents.bar, Frequents.times_a_week FROM Frequents, Frequents AS Frequents_1 WHERE Frequents_1.drinker = 'Bob'"
  },
  {
    id: 'sub6',
    label: 'Sub6',
    question: "Find frequenting records where visit frequency is less than some frequency at 'Pub'.",
    // User input: SELECT * FROM Frequents WHERE times_a_week < SOME (SELECT times_a_week FROM Frequents WHERE bar = 'Pub')
    correctQuery: "SELECT DISTINCT Frequents.drinker, Frequents.bar, Frequents.times_a_week FROM Frequents, Frequents AS Frequents_1 WHERE Frequents.times_a_week < Frequents_1.times_a_week AND Frequents_1.bar = 'Pub'"
  },

  // Case 3: Both Correct and User use EXISTS/SOME
  {
    id: 'sub7',
    label: 'Sub7',
    question: "List all frequenting records for drinkers who have liked at least one beer.",
    // User input: SELECT * FROM Frequents WHERE EXISTS (SELECT * FROM Likes WHERE Likes.drinker = Frequents.drinker)
    correctQuery: "SELECT * FROM Frequents WHERE EXISTS (SELECT * FROM Likes WHERE Likes.drinker = Frequents.drinker)"
  },
  {
    id: 'sub8',
    label: 'Sub8',
    question: "Find all frequenting records where visit frequency is less than some frequency recorded at 'Pub'.",
    // User input: SELECT * FROM Frequents WHERE times_a_week < SOME (SELECT times_a_week FROM Frequents WHERE bar = 'Pub')
    correctQuery: "SELECT * FROM Frequents WHERE times_a_week < SOME (SELECT times_a_week FROM Frequents WHERE bar = 'Pub')"
  },
  {
    id: 'sub9',
    label: 'Sub9',
    question: "Find drinkers who frequent a bar that serves a beer they like.",
    // User input: SELECT DISTINCT drinker FROM Frequents f WHERE EXISTS (SELECT * FROM Serves s, Likes l WHERE s.bar = f.bar AND l.beer = s.beer AND l.drinker = f.drinker)
    correctQuery: "SELECT DISTINCT drinker FROM Frequents f WHERE EXISTS (SELECT * FROM Serves s, Likes l WHERE s.bar = f.bar AND l.beer = s.beer AND l.drinker = f.drinker)"
  },

  // Case 4: Neither uses EXISTS/SOME (standard JOIN queries)
  {
    id: 'sub10',
    label: 'Sub10',
    question: "Find all drinkers and the bars they frequent where they like at least one beer served at that bar.",
    // User input: SELECT DISTINCT F.drinker, F.bar FROM Frequents F, Serves S, Likes L WHERE F.bar = S.bar AND L.drinker = F.drinker AND L.beer = S.beer
    correctQuery: "SELECT DISTINCT F.drinker, F.bar FROM Frequents F, Serves S, Likes L WHERE F.bar = S.bar AND L.drinker = F.drinker AND L.beer = S.beer"
  },

];

// ========== DBLP SCHEMA ==========
// export const dblpQuestions = [
//   // Add DBLP questions here
//   {
//     id: 'q1',
//     label: 'Q1',
//     question: "Find the names of authors, the titles of their conference papers, and the conference names for papers published in 2022.",
//     // User input:
//     correctQuery: "SELECT a.author, i.title, i.booktitle FROM inproceedings i, authorship a WHERE i.pubkey = a.pubkey AND i.yearx = 2022"
//   },
// ];
export const dblpQuestions = [];

// ========== TPC SCHEMA ==========
export const tpcQuestions = [
  // --- Simple WHERE / JOIN ---
  {
    id: 'q1',
    label: 'Q1',
    question: "Find the names and account balances of suppliers located in the nation 'FRANCE'.",
    // User input: SELECT s.s_name, s.s_acctbal FROM supplier s WHERE s.s_nationkey = 7
    // Error: missing nation table, using hardcoded nationkey instead of joining
    correctQuery: "SELECT s.s_name, s.s_acctbal FROM supplier s, nation n WHERE s.s_nationkey = n.n_nationkey AND n.n_name = 'FRANCE'"
  },
  {
    id: 'q2',
    label: 'Q2',
    question: "Find the order keys and customer names for orders with total price greater than 200000.",
    // User input: SELECT o.o_orderkey, c.c_name FROM orders o, customer c WHERE o.o_custkey = c.c_custkey AND o.o_totalprice >= 200000
    // Error: >= instead of >
    correctQuery: "SELECT o.o_orderkey, c.c_name FROM orders o, customer c WHERE o.o_custkey = c.c_custkey AND o.o_totalprice > 200000"
  },
  {
    id: 'q3',
    label: 'Q3',
    question: "Find customer names and order dates for customers in the 'BUILDING' market segment who have orders with status 'O'.",
    // User input: SELECT c.c_name, o.o_orderdate FROM customer c, orders o WHERE c.c_custkey = o.o_custkey AND (c.c_mktsegment = 'BUILDING' OR o.o_orderstatus = 'O')
    // Error: OR instead of AND
    correctQuery: "SELECT c.c_name, o.o_orderdate FROM customer c, orders o WHERE c.c_custkey = o.o_custkey AND c.c_mktsegment = 'BUILDING' AND o.o_orderstatus = 'O'"
  },

  // --- Multi-table JOIN ---
  {
    id: 'q4',
    label: 'Q4',
    question: "Find the names of parts supplied by suppliers in nation 'FRANCE' with supply cost less than 100.",
    // User input: SELECT p.p_name FROM part p, partsupp ps, supplier s WHERE p.p_partkey = ps.ps_partkey AND ps.ps_suppkey = s.s_suppkey AND ps.ps_supplycost < 100
    // Error: missing nation table and nation name filter
    correctQuery: "SELECT p.p_name FROM part p, partsupp ps, supplier s, nation n WHERE p.p_partkey = ps.ps_partkey AND ps.ps_suppkey = s.s_suppkey AND s.s_nationkey = n.n_nationkey AND n.n_name = 'FRANCE' AND ps.ps_supplycost < 100"
  },
  {
    id: 'q5',
    label: 'Q5',
    question: "Find customer names in region 'ASIA' and show their order keys and order priorities.",
    // User input: SELECT c.c_name, o.o_orderkey, o.o_orderpriority FROM customer c, orders o, nation n WHERE c.c_custkey = o.o_custkey AND c.c_nationkey = n.n_nationkey AND n.n_name = 'ASIA'
    // Error: using n_name instead of joining region table and filtering by r_name
    correctQuery: "SELECT c.c_name, o.o_orderkey, o.o_orderpriority FROM customer c, orders o, nation n, region r WHERE c.c_custkey = o.o_custkey AND c.c_nationkey = n.n_nationkey AND n.n_regionkey = r.r_regionkey AND r.r_name = 'ASIA'"
  },

  // --- GROUP BY / HAVING ---
  {
    id: 'q6',
    label: 'Q6',
    question: "For each supplier, find the number of distinct parts they supply.",
    // User input: SELECT s.s_name, COUNT(ps.ps_partkey) FROM supplier s, partsupp ps WHERE s.s_suppkey = ps.ps_suppkey GROUP BY s.s_suppkey
    // Error: GROUP BY s.s_suppkey instead of s.s_name
    correctQuery: "SELECT s.s_name, COUNT(ps.ps_partkey) FROM supplier s, partsupp ps WHERE s.s_suppkey = ps.ps_suppkey GROUP BY s.s_name"
  },
  {
    id: 'q7',
    label: 'Q7',
    question: "Find the names of suppliers who supply more than 10 different parts.",
    // User input: SELECT s.s_name FROM supplier s, partsupp ps WHERE s.s_suppkey = ps.ps_suppkey GROUP BY s.s_name HAVING COUNT(ps.ps_partkey) >= 10
    // Error: >= instead of >
    correctQuery: "SELECT s.s_name FROM supplier s, partsupp ps WHERE s.s_suppkey = ps.ps_suppkey GROUP BY s.s_name HAVING COUNT(ps.ps_partkey) > 10"
  },
  {
    id: 'q8',
    label: 'Q8',
    question: "Find the average supply cost per part for parts that have more than 2 suppliers.",
    // User input: SELECT ps.ps_partkey, SUM(ps.ps_supplycost) FROM partsupp ps GROUP BY ps.ps_partkey HAVING COUNT(ps.ps_suppkey) > 2
    // Error: SUM instead of AVG
    correctQuery: "SELECT ps.ps_partkey, AVG(ps.ps_supplycost) FROM partsupp ps GROUP BY ps.ps_partkey HAVING COUNT(ps.ps_suppkey) > 2"
  },

  // --- Complex WHERE / Self-join ---
  {
    id: 'q9',
    label: 'Q9',
    question: "Find pairs of supplier names that are from the same nation (show each pair only once, ordered by suppkey).",
    // User input: SELECT s1.s_name, s2.s_name FROM supplier s1, supplier s2 WHERE s1.s_nationkey = s2.s_nationkey AND s1.s_suppkey <> s2.s_suppkey
    // Error: <> instead of < (produces duplicate pairs)
    correctQuery: "SELECT s1.s_name, s2.s_name FROM supplier s1, supplier s2 WHERE s1.s_nationkey = s2.s_nationkey AND s1.s_suppkey < s2.s_suppkey"
  },
  {
    id: 'q10',
    label: 'Q10',
    question: "Find line items where the quantity is greater than 30 and either the discount is greater than 0.05 or the tax is less than 0.02.",
    // User input: SELECT l.l_orderkey, l.l_linenumber FROM lineitem l WHERE l.l_quantity > 30 AND l.l_discount > 0.05 AND l.l_tax < 0.02
    // Error: AND instead of OR for the discount/tax condition
    correctQuery: "SELECT l.l_orderkey, l.l_linenumber FROM lineitem l WHERE l.l_quantity > 30 AND (l.l_discount > 0.05 OR l.l_tax < 0.02)"
  },

  // --- Multi-table with GROUP BY + HAVING ---
  {
    id: 'q11',
    label: 'Q11',
    question: "Find customer names in region 'EUROPE' who have placed more than 5 orders. Show the customer name and order count.",
    // User input: SELECT c.c_name, COUNT(o.o_orderkey) FROM customer c, orders o, nation n, region r WHERE c.c_custkey = o.o_custkey AND c.c_nationkey = n.n_nationkey AND n.n_regionkey = r.r_regionkey AND r.r_name = 'EUROPE' GROUP BY c.c_name HAVING COUNT(o.o_orderkey) > 3
    // Error: HAVING threshold > 3 instead of > 5
    correctQuery: "SELECT c.c_name, COUNT(o.o_orderkey) FROM customer c, orders o, nation n, region r WHERE c.c_custkey = o.o_custkey AND c.c_nationkey = n.n_nationkey AND n.n_regionkey = r.r_regionkey AND r.r_name = 'EUROPE' GROUP BY c.c_name HAVING COUNT(o.o_orderkey) > 5"
  },
  {
    id: 'q12',
    label: 'Q12',
    question: "For each order, find the total extended price of its line items. Only show orders whose total extended price exceeds 50000.",
    // User input: SELECT o.o_orderkey, SUM(l.l_extendedprice) FROM orders o, lineitem l WHERE o.o_orderkey = l.l_orderkey GROUP BY o.o_orderkey, o.o_custkey HAVING SUM(l.l_extendedprice) > 50000
    // Error: extra o.o_custkey in GROUP BY
    correctQuery: "SELECT o.o_orderkey, SUM(l.l_extendedprice) FROM orders o, lineitem l WHERE o.o_orderkey = l.l_orderkey GROUP BY o.o_orderkey HAVING SUM(l.l_extendedprice) > 50000"
  },

  // --- Aggregation + complex join ---
  {
    id: 'q13',
    label: 'Q13',
    question: "Find the nation name and the total account balance of all suppliers in that nation, for nations where the total supplier account balance exceeds 1000.",
    // User input: SELECT n.n_name, SUM(s.s_acctbal) FROM supplier s, nation n WHERE s.s_nationkey = n.n_nationkey GROUP BY n.n_name HAVING AVG(s.s_acctbal) > 1000
    // Error: AVG in HAVING instead of SUM
    correctQuery: "SELECT n.n_name, SUM(s.s_acctbal) FROM supplier s, nation n WHERE s.s_nationkey = n.n_nationkey GROUP BY n.n_name HAVING SUM(s.s_acctbal) > 1000"
  },
  {
    id: 'q14',
    label: 'Q14',
    question: "Find part names and their brand for parts with retail price greater than 1500 that are supplied by at least one supplier with account balance greater than 5000.",
    // User input: SELECT p.p_name, p.p_brand FROM part p, partsupp ps, supplier s WHERE p.p_partkey = ps.ps_partkey AND ps.ps_suppkey = s.s_suppkey AND p.p_retailprice > 1500
    // Error: missing supplier account balance condition
    correctQuery: "SELECT p.p_name, p.p_brand FROM part p, partsupp ps, supplier s WHERE p.p_partkey = ps.ps_partkey AND ps.ps_suppkey = s.s_suppkey AND p.p_retailprice > 1500 AND s.s_acctbal > 5000"
  },
  {
    id: 'q15',
    label: 'Q15',
    question: "Find the names of suppliers who supply parts with a supply cost greater than 500 and also supply parts with a supply cost less than 50.",
    // User input: SELECT s.s_name FROM supplier s, partsupp ps WHERE s.s_suppkey = ps.ps_suppkey AND ps.ps_supplycost > 500 AND ps.ps_supplycost < 50
    // Error: should use two partsupp aliases, not AND on same row
    correctQuery: "SELECT s.s_name FROM supplier s, partsupp ps1, partsupp ps2 WHERE s.s_suppkey = ps1.ps_suppkey AND s.s_suppkey = ps2.ps_suppkey AND ps1.ps_supplycost > 500 AND ps2.ps_supplycost < 50"
  },

  // --- Date-related queries ---
  {
    id: 'q16',
    label: 'Q16',
    question: "Find order keys, customer names, and order dates for orders placed after '1996-01-01' with total price greater than 100000.",
    // User input: SELECT o.o_orderkey, c.c_name, o.o_orderdate FROM orders o, customer c WHERE o.o_custkey = c.c_custkey AND o.o_orderdate > DATE '1996-01-01'
    // Error: missing total price condition
    correctQuery: "SELECT o.o_orderkey, c.c_name, o.o_orderdate FROM orders o, customer c WHERE o.o_custkey = c.c_custkey AND o.o_orderdate > DATE '1996-01-01' AND o.o_totalprice > 100000"
  },
  {
    id: 'q17',
    label: 'Q17',
    question: "Find line items that were shipped late (ship date is after commit date). Show the order key, line number, ship date, and commit date.",
    // User input: SELECT l.l_orderkey, l.l_linenumber, l.l_shipdate, l.l_commitdate FROM lineitem l WHERE l.l_shipdate >= l.l_commitdate
    // Error: >= instead of > (equal means on time, not late)
    correctQuery: "SELECT l.l_orderkey, l.l_linenumber, l.l_shipdate, l.l_commitdate FROM lineitem l WHERE l.l_shipdate > l.l_commitdate"
  },
  {
    id: 'q18',
    label: 'Q18',
    question: "Find customer names and the number of their orders placed between '1995-01-01' and '1996-12-31' (inclusive). Only show customers with more than 3 orders in that period.",
    // User input: SELECT c.c_name, COUNT(o.o_orderkey) FROM customer c, orders o WHERE c.c_custkey = o.o_custkey AND o.o_orderdate >= DATE '1995-01-01' AND o.o_orderdate <= DATE '1996-12-31' GROUP BY c.c_name HAVING COUNT(o.o_orderkey) >= 3
    // Error: HAVING >= 3 instead of > 3
    correctQuery: "SELECT c.c_name, COUNT(o.o_orderkey) FROM customer c, orders o WHERE c.c_custkey = o.o_custkey AND o.o_orderdate >= DATE '1995-01-01' AND o.o_orderdate <= DATE '1996-12-31' GROUP BY c.c_name HAVING COUNT(o.o_orderkey) > 3"
  },
  {
    id: 'q19',
    label: 'Q19',
    question: "Find supplier names who have line items shipped after '1997-01-01' with quantity greater than 20. Show each supplier name only once.",
    // User input: SELECT DISTINCT s.s_name FROM supplier s, lineitem l WHERE s.s_suppkey = l.l_suppkey AND l.l_shipdate > DATE '1997-01-01'
    // Error: missing quantity condition
    correctQuery: "SELECT DISTINCT s.s_name FROM supplier s, lineitem l WHERE s.s_suppkey = l.l_suppkey AND l.l_shipdate > DATE '1997-01-01' AND l.l_quantity > 20"
  },
  {
    id: 'q20',
    label: 'Q20',
    question: "For each nation, find the total extended price of line items shipped before '1996-01-01' by suppliers in that nation. Only show nations with total extended price exceeding 500000.",
    // User input: SELECT n.n_name, SUM(l.l_extendedprice) FROM nation n, supplier s, lineitem l WHERE n.n_nationkey = s.s_nationkey AND s.s_suppkey = l.l_suppkey AND l.l_shipdate < DATE '1996-01-01' GROUP BY n.n_name HAVING SUM(l.l_extendedprice) > 500000 AND l.l_receiptdate < DATE '1996-01-01'
    // Error: extra condition l.l_receiptdate in HAVING (wrong clause placement, should be in WHERE if needed)
    correctQuery: "SELECT n.n_name, SUM(l.l_extendedprice) FROM nation n, supplier s, lineitem l WHERE n.n_nationkey = s.s_nationkey AND s.s_suppkey = l.l_suppkey AND l.l_shipdate < DATE '1996-01-01' GROUP BY n.n_name HAVING SUM(l.l_extendedprice) > 500000"
  },
];

// Map schema id to questions
export const questionsBySchema = {
  beers: beersQuestions,
  dblp: dblpQuestions,
  tpc: tpcQuestions
};

// Default export for backward compatibility
export const questionsData = beersQuestions;
