\COPY Bar(name, address) FROM 'data/Bar.dat' WITH DELIMITER ',' NULL '' CSV
\COPY Beer(name, brewer) FROM 'data/Beer.dat' WITH DELIMITER ',' NULL '' CSV
\COPY Drinker(name, address) FROM 'data/Drinker.dat' WITH DELIMITER ',' NULL '' CSV
\COPY Frequents(drinker, bar, times_a_week) FROM 'data/Frequents.dat' WITH DELIMITER ',' NULL '' CSV
\COPY Serves(bar, beer, price) FROM 'data/Serves.dat' WITH DELIMITER ',' NULL '' CSV
\COPY Likes(drinker, beer) FROM 'data/Likes.dat' WITH DELIMITER ',' NULL '' CSV
