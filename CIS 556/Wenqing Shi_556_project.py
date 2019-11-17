import requests
from lxml import etree
import pymysql
from decimal import Decimal

# Connect to the database
connection = pymysql.connect(host='127.0.0.1',
                             user='root',
                             password='wenqing',
                             db='SceneryReview',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

cursorObject = connection.cursor()


# SQL query string
sqlQuery = "drop table if exists Users"
cursorObject.execute(sqlQuery)

sqlQuery = "create table Users (USERID varchar(255) Not Null, address varchar(120), primary key (USERID))"
cursorObject.execute(sqlQuery)

sqlQuery = "drop table if exists Reviews"
cursorObject.execute(sqlQuery)

sqlQuery = "create table Reviews (USERID varchar(255) Not Null, RATINGDATE varchar(120) Not Null, attractionname varchar(120), viamobile varchar(30), ratingscore int, reviewtitle varchar(1000), primary key(USERID, RATINGDATE))"
cursorObject.execute(sqlQuery)

sqlQuery = "drop table if exists Attractions"
cursorObject.execute(sqlQuery)

sqlQuery = "create table Attractions(CITYID varchar(255) Not Null, ATTRACTIONNAME varchar(120) Not Null, attractiontype varchar(60), STREET varchar(60), phone varchar(30), reviewnum varchar(30), attractionscore int, primary key(CITYID, ATTRACTIONNAME, STREET))"
cursorObject.execute(sqlQuery)

sqlQuery = "drop table if exists Cities"
cursorObject.execute(sqlQuery)

sqlQuery = "create table Cities (CITYID varchar(120) NOT Null, cityname varchar(30), country varchar(30), state varchar(30), continent varchar(30), capital int, primary key (CITYID))"
cursorObject.execute(sqlQuery)

sqlQuery = "drop table if exists Nearby_hotel"
cursorObject.execute(sqlQuery)

sqlQuery = "create table Nearby_hotel (CITYID varchar(120) NOT Null, ATTRACTIONNAME varchar(120) Not Null, NBHOTEL_NAME varchar(120) Not Null, nbhotel_reviewnum varchar(30), nbhotel_mile varchar(30), primary key(CITYID, ATTRACTIONNAME, NBHOTEL_NAME))"
cursorObject.execute(sqlQuery)

sqlQuery = "drop table if exists Nearby_restaurant"
cursorObject.execute(sqlQuery)

sqlQuery = "create table Nearby_restaurant (CITYID varchar(120) NOT Null, ATTRACTIONNAME varchar(120) Not Null, NBRESTAURANT_NAME varchar(120) Not Null, nbrestaurant_reviewnum varchar(30), nbrestaurant_mile varchar(30), primary key(CITYID, ATTRACTIONNAME, NBRESTAURANT_NAME))"
cursorObject.execute(sqlQuery)


sqlQuery = "drop table if exists Nearby_attraction"
cursorObject.execute(sqlQuery)

sqlQuery = "create table Nearby_attraction (CITYID varchar(120) NOT Null, ATTRACTIONNAME varchar(120) Not Null, NBATTRACTION_NAME varchar(120) Not Null, nbattraction_reviewnum varchar(30), nbattraction_mile varchar(30), primary key(CITYID, ATTRACTIONNAME, NBATTRACTION_NAME))"
cursorObject.execute(sqlQuery)

try:
    sqlQuery = "insert into Cities values('" + "0001" + "','" + "Detroit" + "','" + "the_United_States" + "','" + "Michigan" + "','" + "America" + "','" + "1" + "')"
    cursorObject.execute(sqlQuery)
    connection.commit()
except pymysql.err.IntegrityError as e:
    print (e)
except pymysql.err.ProgrammingError as e:
    print (e)

#This three line defined what scenery you want to choose, url is the full entry path
Detroit_url = 'https://www.tripadvisor.com/Attractions-g42139-Activities-Detroit_Michigan.html#FILTERED_LIST'
first_part_detroit_url = "https://www.tripadvisor.com/Attractions-g42139-Activities-oa"
last_part_detroit_url = "-Detroit_Michigan.html#FILTERED_LIST"

sgwbdata = requests.get(Detroit_url)
sgwbdata.encoding = 'utf8mb4'
sgtree = etree.HTML((sgwbdata.text))
max2=0
for nextsgattractionurlLink in sgtree.xpath("//div[@class='pageNumbers']//a/text()"):
    if(int(nextsgattractionurlLink)>max2):
        max2 = int(nextsgattractionurlLink)
print(max2)
sgattractionurlList=[]
sgattractionurlList.append(Detroit_url)
print(sgattractionurlList[0])

for m in range(1,max2):
    index2 = str(m*30)
    # print(first_part_sgattractionurl+index2+last_part_sgattractionurl)
    sgattractionurlList.append(first_part_detroit_url + index2 + last_part_detroit_url)
    print(sgattractionurlList[m])
    print("Comment index2:" + str(m))

all_scenery = []
for m in range(0,len(sgattractionurlList)):
    page2 = requests.get(sgattractionurlList[m])
    page2.encoding = 'utf8mb4'
    sgtree = etree.HTML(page2.text)

    for n in range(0,30):
        try:
            scenery = str(sgtree.xpath("//*[@id='ATTR_ENTRY_']/div[2]/div/div/div[1]/div[2]/a/@href")[n])
            print("Attraction url:" + scenery)
            all_scenery.append("https://www.tripadvisor.com"+scenery)
        except IndexError:
          print("Singleactionurl url is missing!")

for p in range(0,len(all_scenery)):

    urlcode = str(all_scenery[p]).split("Review")[1].split("Reviews")[0]
    sceneryname =  str(all_scenery[p]).split("Reviews")[1]
    new_sceneryname = sceneryname.replace("-", "").replace(".html", "")
    print("urlcode" + urlcode)
    print("new_sceneryname" + new_sceneryname)
    url = "https://www.tripadvisor.com/Attraction_Review"+urlcode+"Reviews"+sceneryname
    first_part_url = "https://www.tripadvisor.com/Attraction_Review"+urlcode+"Reviews-or"
    last_part_url = "-"+new_sceneryname
    wbdata = requests.get(all_scenery[p])
    wbdata.encoding = 'utf8mb4'
    tree = etree.HTML((wbdata.text))
    max=0
    for nextLink in tree.xpath("//div[@class='pageNumbers']//a/@data-page-number"):
       if(int(nextLink)>max):
           max = int(nextLink)
    print(max)
    websiteList=[]
    websiteList.append("https://www.tripadvisor.com/Attraction_Review"+urlcode+"Reviews"+sceneryname)
    print(websiteList[0])
    # After understand their algorithm, generate all the comment list
    for i in range(1,max):
       index = str(i*10)
       websiteList.append(first_part_url+index+last_part_url)
       print("website list:   "+websiteList[i])

    # Open each link and grab the information
    for i in range(0,len(websiteList)):
       page = requests.get(websiteList[i])
       page.encoding = 'utf8mb4'
       tree = etree.HTML(page.text)
       print("Comment index:"+str(i))

       if i==0:

           try:
               attractionname = str(tree.xpath("//div[@data-placement-name='location_detail_header:attractions']//h1//text()")[i])
               print("attractionname :" + attractionname)
           except IndexError:
               reviewnum = "null"
               print("attractionname is missing!")
           try:
               reviewnum = str(tree.xpath("//div[@class='rs rating']//a[@class='more']//span/text()")[i])
               print("Reviewnum :" + reviewnum)
           except IndexError:
               reviewnum = "null"
               print("Reviewnum is missing!")
           try:
               attractionscore = str(tree.xpath("//div[@class='prw_rup prw_common_bubble_rating bubble_rating']//span[@property='ratingValue']/@content")[i])
               print("Attractionscore :" + attractionscore)
           except IndexError:
               attractionscore = "0"
               print("Attractionscore is missing!")
           try:
               street = str(tree.xpath("//*[@id='taplc_location_detail_header_attractions_0']/div[3]/div/div[1]/span[2]//text()")[i])
               print("street :" + street)
           except IndexError:
               street = "null"
               print("street is missing!")
           try:
               phone = str(tree.xpath("// *[ @ id = 'taplc_location_detail_header_attractions_0'] / div[3] / div / div[2] / span[2]/text()")[i])
               print("phone :" + phone)
           except IndexError:
               phone = "null"
               print("phone is missing!")
           try:
               attractiontype = str(tree.xpath("//*[@id='taplc_location_detail_header_attractions_0']/div[2]/span[3]/div/a[1]//text()")[i])
               print("attractiontype :" + attractiontype)
           except IndexError:
               attractiontype = "null"
               print("attractiontype is missing!")
           try:
               nbhotel_name = str(tree.xpath("// *[ @ id = 'LOCATION_TAB'] / div[3] / div / div[1] / div / div[2] / div[1]/text()")[i])
               print("nbhotel_name :" + nbhotel_name)
           except IndexError:
               nbhotel_name = "null"
               print("nbhotel_name is missing!")
           try:
               nbhotel_mile = str(tree.xpath("//*[@id='LOCATION_TAB']/div[3]/div/div[1]/div/div[2]/div[4]/text()")[i])
               print("nbhotel_mile :" + nbhotel_mile)
           except IndexError:
               nbhotel_mile = "null"
               print("nbhotelmile is missing!")
           try:
               nbhotel_reviewnum = str(tree.xpath("// *[ @ id = 'LOCATION_TAB'] / div[3] / div / div[1] / div / div[2] / div[3]/text()")[i])
               print("nbhotel_reviewnum :" + nbhotel_reviewnum)
           except IndexError:
               nbhotel_reviewnum = "null"
               print("nbhotel_reviewnum is missing!")
           try:
               nbrestaurant_name = str(tree.xpath("//*[@id='LOCATION_TAB']/div[4]/div/div[1]/div/div[2]/div[1]/text()")[i])
               print("nbrestaurant_name :" + nbrestaurant_name)
           except IndexError:
               nbrestaurant_name = "null"
               print("nbrestaurant_name is missing!")
           try:
               nbrestaurant_mile = str(tree.xpath("//*[@id='LOCATION_TAB']/div[4]/div/div[1]/div/div[2]/div[4]/text()")[i])
               print("nbrestaurant_mile :" + nbrestaurant_mile)
           except IndexError:
               nbrestaurant_mile = "null"
               print("nbrestaurant_mile is missing!")
           try:
               nbrestaurant_reviewnum = str(tree.xpath("//*[@id='LOCATION_TAB']/div[4]/div/div[3]/div/div[2]/div[3]/text()")[i])
               print("nbrestaurant_reviewnum :" + nbrestaurant_reviewnum)
           except IndexError:
               nbrestaurant_reviewnum = "null"
               print("nbrestaurant_reviewnum is missing!")
           try:
               nbattraction_name = str(tree.xpath("//*[@id='LOCATION_TAB']/div[5]/div/div[1]/div/div[2]/div[1]/text()")[i])
               print("nbattraction_name :" + nbattraction_name)
           except IndexError:
               nbattraction_name = "null"
               print("nbattraction_name is missing!")
           try:
               nbattraction_mile = str(tree.xpath("//*[@id='LOCATION_TAB']/div[5]/div/div[1]/div/div[2]/div[4]/text()")[i])
               print("nbattraction_mile :" + nbattraction_mile)
           except IndexError:
               nbattraction_mile = "null"
               print("nyrestaurant_mile is missing!")
           try:
               nbattraction_reviewnum = str(tree.xpath("//*[@id='LOCATION_TAB']/div[5]/div/div[1]/div/div[2]/div[3]/text()")[i])
               print("nbattraction_reviewnum :" + nbattraction_reviewnum)
           except IndexError:
               nbattraction_reviewnum = "null"
               print("nbattraction_reviewnum is missing!")
           try:
               sqlQuery = "insert into Attractions values('" + "0001" + "','" + attractionname + "','" + attractiontype + "','" + street + "','" + phone + "','" + reviewnum + "','" + attractionscore + "')"
               cursorObject.execute(sqlQuery)
           except pymysql.err.IntegrityError as e:
               print (e)
           except pymysql.err.ProgrammingError as e:
               print (e)
           try:
               sqlQuery = "insert into Nearby_hotel values('" + "0001" + "','" + attractionname + "','" + nbhotel_name + "','" + nbhotel_reviewnum + "','" + nbhotel_mile + "')"
               cursorObject.execute(sqlQuery)
           except pymysql.err.IntegrityError as e:
               print (e)
           except pymysql.err.ProgrammingError as e:
               print (e)
           try:
               sqlQuery = "insert into Nearby_restaurant values('" + "0001" + "','" + attractionname + "','" + nbrestaurant_name + "','" + nbrestaurant_reviewnum + "','" + nbrestaurant_mile + "')"
               cursorObject.execute(sqlQuery)
           except pymysql.err.IntegrityError as e:
               print (e)
           except pymysql.err.ProgrammingError as e:
               print (e)
           try:
               sqlQuery = "insert into Nearby_attraction values('" + "0001" + "','" + attractionname + "','" + nbattraction_name + "','" + nbattraction_reviewnum + "','" + nbattraction_mile + "')"
               cursorObject.execute(sqlQuery)
           except pymysql.err.IntegrityError as e:
               print (e)
           except pymysql.err.ProgrammingError as e:
               print (e)
           connection.commit()


       # for user in tree.xpath("//div[@class='mainContent']/div[@class='innerBubble']/div[@class='wrap']"):
       user = tree.xpath("//div[@class='mainContent']/div[@class='innerBubble']/div[@class='wrap']")
       for j in range(0, len(user)):
            print("-------------------Next User Info------------------")
            userid = ""
            useraddr = ""
            ratingdate = ""
            viamobile = ""
            reviewtitle =""
            ratingscore = ""

            try:
               userid = user[j].xpath("//div[@class='userInfo']//span[@class='expand_inline scrname']/text()")[j]
               print("User name:" + userid)
              # add user name into your database
            except IndexError:
                userid = "null"
                print("User name is missing!")
            try:
                useraddr = user[j].xpath("//span[@class='location']//span[@class='expand_inline userLocation']//text()")[j]
                print("User's address:" + useraddr)
            except IndexError:
                useraddr = "null"
                print("User address is missing!")
            try:
                ratingdate = user[j].xpath("//div[@class='rating reviewItemInline']//span[@class='ratingDate relativeDate']//text()")[j]
                print("User's rating data:" + ratingdate)
            except IndexError:
                ratingdate = "0"
                print("User rating date is missing!")
            try:
                viamobile = user[j].xpath("//div[@class='viaMobile']//span[@class='ui_icon mobile-phone']//text()")[j]
                viamobile = "Yes"
                print("User via mobile:" + viamobile)
            except IndexError:
                viamobile = "No"
                print("User via mobile not!")
            try:
                reviewtitle = user[j].xpath("//span[@class='noQuotes']/text()")[j].replace("'","&rsquo").replace('"',"&rdquo")
                print("User review title:" + reviewtitle)
            except IndexError:
                reviewtitle= "null"
                print("User review title not!")
            except pymysql.err.InternalError as e:
                code, msg = e.args
                if code == 1050:
                    print('tblName  '+ 'already exists')
            try:
                old_ratingscore = user[j].xpath("//div[@class='ratingInfo']//span[1]/@class")[j]
                ratingscore = old_ratingscore.split("_")[len(old_ratingscore.split("_"))-1]
                print("User review score: "+ str(Decimal(ratingscore)/10))
            except IndexError:
                ratingscore="null"
                print("User review score not!")
            try:
                sqlQuery = "insert into Users values('"+userid+"','"+useraddr+"')"
                cursorObject.execute(sqlQuery)
            except pymysql.err.IntegrityError as e:
                print (e)
            except pymysql.err.ProgrammingError as e:
                print (e)
            try:

                sqlQuery = "insert into Reviews values('"+userid+"','" +str(ratingdate)+ \
                           "','" +attractionname+ "','" +viamobile+ "','" +str(Decimal(ratingscore)/10)+ "','"+str(reviewtitle)+"')"
                cursorObject.execute(sqlQuery)
            except pymysql.err.IntegrityError as e:
                print (e)
            except pymysql.err.ProgrammingError as e:
                print (e)
            except pymysql.err.InternalError as e:
                reviewtitle="Incorrect string value"
                print (e)
                try:
                    sqlQuery = "insert into Reviews values('" + userid + "','" + str(ratingdate) + \
                               "','" + attractionname + "','" + viamobile + "','" + str(
                        Decimal(ratingscore) / 10) + "','" + reviewtitle + "')"
                    cursorObject.execute(sqlQuery)
                except IndexError as e:
                    print (e)
                except pymysql.err.IntegrityError as e:
                    print(e)


            connection.commit()





sqlQuery = "show tables"
cursorObject.execute(sqlQuery)

sqlQuery = "select * from Users limit 100"
cursorObject.execute(sqlQuery)

# Fetch all the rows

rows = cursorObject.fetchall()

for row in rows:
    print(row)

connection.close()



