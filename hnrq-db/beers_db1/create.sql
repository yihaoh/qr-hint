CREATE TABLE Bar(name VARCHAR(70) NOT NULL PRIMARY KEY,
                 address VARCHAR(70));
CREATE TABLE Beer(name VARCHAR(70) NOT NULL PRIMARY KEY,
                  brewer VARCHAR(70));
CREATE TABLE Drinker(name VARCHAR(70) NOT NULL PRIMARY KEY,
                     address VARCHAR(70));
CREATE TABLE Frequents(drinker VARCHAR(70) NOT NULL REFERENCES Drinker(name),
                       bar VARCHAR(70) NOT NULL REFERENCES Bar(name),
                       times_a_week SMALLINT CHECK(times_a_week > 0),
                       PRIMARY KEY(drinker, bar));
CREATE TABLE Serves(bar VARCHAR(70) NOT NULL REFERENCES Bar(name),
                    beer VARCHAR(70) NOT NULL REFERENCES Beer(name),
                    price DECIMAL(5,2) CHECK(price > 0),
                    PRIMARY KEY(bar, beer));
CREATE TABLE Likes(drinker VARCHAR(70) NOT NULL REFERENCES Drinker(name),
                   beer VARCHAR(70) NOT NULL REFERENCES Beer(name),
                   PRIMARY KEY(drinker, beer));
