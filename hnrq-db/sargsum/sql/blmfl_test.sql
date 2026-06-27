-- -- Set the bitsize
SET blmfl.bloomfilter_bitsize TO 256;
SET blmfl.estimated_count TO 10;
SET blmfl.num_hashes TO 15;

DROP TABLE R;
DROP TABLE S;
DROP TABLE T;
DROP TABLE X;
DROP TABLE Y;
DROP TABLE Z;

DROP TABLE blmfl_R;
DROP TABLE blmfl_S;
DROP TABLE blmfl_T;
DROP TABLE blmfl_X;
DROP TABLE blmfl_Y;
DROP TABLE blmfl_Z;

DROP TABLE blmfl_merged;

DROP EXTENSION blmfl;

-- Create the extension
CREATE EXTENSION blmfl;

-- Table X
-- Test "ANY" with (INT, INT)
CREATE TABLE X(A INT, B INT);
CREATE TABLE blmfl_X(summary BLMFL_RESULT);

INSERT INTO X VALUES (5, 24), (2, 53), (90, 1), (12, 56);
INSERT INTO blmfl_X (SELECT blmfl_any(A, B) FROM X);
SELECT * FROM blmfl_X;

SELECT blmfl_test_any((SELECT summary FROM blmfl_X), 5, 24);    -- True
SELECT blmfl_test_any((SELECT summary FROM blmfl_X), 5, 12);    -- False
SELECT blmfl_test_any((SELECT summary FROM blmfl_X), 90, 1);    -- True

-- Table Y
-- Test "ANY" with (BIGINT, INT)
CREATE TABLE Y(A BIGINT, B INT);
CREATE TABLE blmfl_Y(summary BLMFL_RESULT);

INSERT INTO Y VALUES (3, 2), (6223372036854775807, 12), (9, 76), (9223372036000775807, 0);
INSERT INTO blmfl_Y (SELECT blmfl_any(A, B) FROM Y);
SELECT * FROM blmfl_Y;

SELECT blmfl_test_any((SELECT summary FROM blmfl_Y), 9, 76);                     -- True
SELECT blmfl_test_any((SELECT summary FROM blmfl_Y), 6223372036854775807, 100);  -- False
SELECT blmfl_test_any((SELECT summary FROM blmfl_Y), 9223372036000775807, 0);    -- True

-- Table Z
-- Test "ANY" with (VARCHAR(n), CHAR(n))
CREATE TABLE Z(A VARCHAR(30), B CHAR(3));
CREATE TABLE blmfl_Z(summary BLMFL_RESULT);

INSERT INTO Z VALUES ('heLLO', 'yes'), ('Byee', 'no'), ('Dogs!', 'idk');
INSERT INTO blmfl_Z (SELECT blmfl_any(A, B) FROM Z);
SELECT * FROM blmfl_Z;

SELECT blmfl_test_any((SELECT summary FROM blmfl_Z), CAST('heLLO' AS VARCHAR(30)), CAST('yes' AS CHAR(3)));      -- True
SELECT blmfl_test_any((SELECT summary FROM blmfl_Z), CAST('hello there' AS VARCHAR(30)), CAST('NOO' AS CHAR(3)));      -- False
SELECT blmfl_test_any((SELECT summary FROM blmfl_Z), CAST('Dogs!' AS VARCHAR(30)), CAST('idk' AS CHAR(3)));      -- True

-- Table R
-- Test "BYTEA" with converted FLOAT and VARCHAR(n)
CREATE TABLE R(A BYTEA, B BYTEA);
CREATE TABLE blmfl_R(summary BLMFL_RESULT);

INSERT INTO R VALUES (numeric_send(253.64), textsend('Hello')), (numeric_send(123.45), textsend('Bye')), (numeric_send(3562.1), textsend('Yay')), (numeric_send(9582.3), textsend('Tests'));
INSERT INTO blmfl_R (SELECT blmfl(A, B) FROM R);
SELECT * FROM blmfl_R;

SELECT blmfl_test((SELECT summary FROM blmfl_R), numeric_send(253.64), textsend('Invalid'));         -- False 
SELECT blmfl_test((SELECT summary FROM blmfl_R), numeric_send(253.64), textsend('Hello'));           -- True
SELECT blmfl_test((SELECT summary FROM blmfl_R), numeric_send(123.45), textsend('Bye'));             -- True
SELECT blmfl_test((SELECT summary FROM blmfl_R), numeric_send(9582.3), textsend('Tests'));           -- True
SELECT blmfl_test((SELECT summary FROM blmfl_R), numeric_send(123.45), textsend('bye'));             -- False ('B'' should be Uppercase)

-- Table S 
-- Test "BYTEA" with converted INT
CREATE TABLE S(C BYTEA);
CREATE TABLE blmfl_S(summary BLMFL_RESULT);

INSERT INTO S VALUES (numeric_send(253464)), (numeric_send(5654635)), (numeric_send(1342534)), (numeric_send(352465)), (numeric_send(473657));
INSERT INTO blmfl_S (SELECT blmfl(C) FROM S);
SELECT * FROM blmfl_S;

SELECT blmfl_test((SELECT summary FROM blmfl_S), numeric_send(253));         -- False
SELECT blmfl_test((SELECT summary FROM blmfl_S), numeric_send(253464));      -- True
SELECT blmfl_test((SELECT summary FROM blmfl_S), numeric_send(1342534));     -- True
SELECT blmfl_test((SELECT summary FROM blmfl_S), numeric_send(473657));      -- True

-- Table T 
-- Test "BYTEA" with converted INT
CREATE TABLE T(D BYTEA);
CREATE TABLE blmfl_T(summary BLMFL_RESULT);

SET blmfl.estimated_count TO 3;
INSERT INTO T VALUES (numeric_send(1)), (numeric_send(2)), (numeric_send(3));
INSERT INTO blmfl_T (SELECT blmfl(D) FROM T);
SELECT * FROM blmfl_T;

-- Test other functions
SELECT blmfl_optimal_k(64, 4);
SELECT blmfl_fpr((SELECT summary FROM blmfl_R));    -- Need a double parenthesis here

-- Try to merge S and T's bloomfilters
CREATE TABLE blmfl_merged(summary BLMFL_RESULT);
INSERT INTO blmfl_merged (SELECT blmfl_merge((SELECT summary FROM blmfl_S), (SELECT summary FROM blmfl_T)));
SELECT summary FROM blmfl_merged;

SELECT blmfl_test((SELECT summary FROM blmfl_merged), numeric_send(253464));      -- True
SELECT blmfl_test((SELECT summary FROM blmfl_merged), numeric_send(1));           -- True
SELECT blmfl_test((SELECT summary FROM blmfl_merged), numeric_send(100));         -- False

-- Cleanup
DROP TABLE R;
DROP TABLE S;
DROP TABLE T;
DROP TABLE X;
DROP TABLE Y;
DROP TABLE Z;

DROP TABLE blmfl_R;
DROP TABLE blmfl_S;
DROP TABLE blmfl_T;
DROP TABLE blmfl_X;
DROP TABLE blmfl_Y;
DROP TABLE blmfl_Z;

DROP TABLE blmfl_merged;

DROP EXTENSION blmfl;