\COPY Bar(name, address) FROM 'data/bar.txt' WITH DELIMITER ',' NULL '' CSV HEADER
\COPY Beer(name, brewer) FROM 'data/beer.txt' WITH DELIMITER ',' NULL '' CSV HEADER
\COPY Drinker(name, address) FROM 'data/drinker.txt' WITH DELIMITER ',' NULL '' CSV HEADER
\COPY Frequents(drinker, bar, times_a_week) FROM 'data/frequents.txt' WITH DELIMITER ',' NULL '' CSV HEADER
\COPY Serves(bar, beer, price) FROM 'data/serves.txt' WITH DELIMITER ',' NULL '' CSV HEADER
\COPY Likes(drinker, beer) FROM 'data/likes.txt' WITH DELIMITER ',' NULL '' CSV HEADER
