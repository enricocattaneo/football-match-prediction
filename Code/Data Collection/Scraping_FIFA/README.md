# FIFA Data Scraper
## For Both Players and Teams Data

**AIM**: Using the BeautifulSoup package, scrape football teams' and players' information about FIFA attributes from the fifaindex.com website. This code considers from 2011 onwards and for five major European leagues (but can easily be extended to other years and football competitions).

Example of personally derived datasets: https://www.kaggle.com/datasets/enricocattaneo/fifa-videogame-players-and-teams-data.

# **Scrape Teams & Scrape Players**

- [All Functions](#allfunctions)
- [To Use](#touse)
- [Final Considerations](#consideration)
- [Target Data Examples](#examples)


## **All Functions** <a id='allfunctions'></a>

We needed these functions to transform the input information into a usable form and retrieve that information.
* **make_request()**: necessary to make calls using different headers with a sleep time between each call (good practice) and handle some possible exceptions when requesting to a website.  
* **get_allpages()**: given a list of seasons and leagues to consider, it creates the URL of the fifaindex.com wanted webpage to later scrape information from (concatenate together the different components in the URL structure).
* **get_teamlinks()**: given a URL, it returns all teams' links in it (first scraping function).
* **scrape_team()**: Given a specific team webpage (obtained from the previous function), returns the desired (team's attributes) information using the BeautifulSoup package.
* **Open_Create_file()**: given season and leagues lists, it uses get_allpages() to return an object with every team link (also checks if such an object already exists in our directory).
* **get_final_list()**: given all teams' page links (from the previous function), it returns the final object containing info on each wanted team.
* **final_dataframe()**: it just chains the two previous functions and their output, returning a Dataframe.

## **To Use** <a id='touse'></a>

Find the seasons and leagues you want to scrape data of. Like in the image:


![1](https://user-images.githubusercontent.com/80990030/182503848-9d221477-607b-43f2-9805-2c8848e71343.png)


Put the desired leagues and seasons in two lists in the same format as they appear in the image and the code. Run the code to obtain a Dataframe (with the possibility to save it as a CSV file).  

## Final Considerations <a id='consideration'></a>

*The code could take a long time to run due to the sleep time in the make_request() function (better to keep at least some).*

*The same considerations and instructions are worth for the player data scraper (functions names are similar but might be different). Also, you should build the seasons and leagues lists likewise. In fact, (as shown in the following image) players' page URLs and teams' page URLs are constructed similarly.*

<img width="1418" alt="2" src="https://user-images.githubusercontent.com/80990030/182503903-0f6fbc00-48ea-4ceb-8592-ca975257ec44.png">


## Target Data Examples <a id='examples'></a>
**Team Example**

![3](https://user-images.githubusercontent.com/80990030/182503932-cb8cc5ca-2b4e-4823-89fc-e8e7b43335bb.png)

**Player Example**

![4](https://user-images.githubusercontent.com/80990030/182503961-75981ec1-4b4a-43bc-991c-f98641058da4.png)
