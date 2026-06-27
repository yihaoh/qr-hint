CREATE TABLE Bar(name VARCHAR(20) NOT NULL PRIMARY KEY,
                 address VARCHAR(20));
CREATE TABLE Beer(name VARCHAR(20) NOT NULL PRIMARY KEY,
                  brewer VARCHAR(20));
CREATE TABLE Drinker(name VARCHAR(20) NOT NULL PRIMARY KEY,
                     address VARCHAR(20));
CREATE TABLE Frequents(drinker VARCHAR(20) NOT NULL REFERENCES Drinker(name),
                       bar VARCHAR(20) NOT NULL REFERENCES Bar(name),
                       times_a_week SMALLINT CHECK(times_a_week > 0),
                       PRIMARY KEY(drinker, bar));
CREATE TABLE Serves(bar VARCHAR(20) NOT NULL REFERENCES Bar(name),
                    beer VARCHAR(20) NOT NULL REFERENCES Beer(name),
                    price DECIMAL(5,2) CHECK(price > 0),
                    PRIMARY KEY(bar, beer));
CREATE TABLE Likes(drinker VARCHAR(20) NOT NULL REFERENCES Drinker(name),
                   beer VARCHAR(20) NOT NULL REFERENCES Beer(name),
                   PRIMARY KEY(drinker, beer));
