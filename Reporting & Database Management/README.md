# Managing Customer and Product Data
The following project contains my solutions to the SQL (T-SQL) challenges hosted on the Microsoft Learn framework. The DBMS used for this project is Azure SQL Database, with the project utilising a sample DB for the fictitious _Adventure Works Cycles_ company whose tables (and foreign key references) are shown below:

![adventureworks-erd](https://github.com/WilliamBaxter417/Portfolio/blob/main/Reporting%20%26%20Database%20Management/adventureworks-erd.png)

To generate the sample DB, use the following instructions. _Note: these instructions are taken from the Microsoft Learn resource where Azure Data Studio is used as the main program._
1. Download the **adventureworkslt.sql** script (included in this repository).
2. Start Azure Data Studio, and open the **adventureworkslt.sql** script file.
3. In the script pane, connect to your SQL Server Express server using the following information:
    - **Connection type**: SQL Server
    - **Server**: localhost\SQLExpress
    - **Authentication type**: Windows Authentication
    - **Database**: master
    - **Server group**: \<Default\>
    - **Name**: _leave blank_
4. Ensure the **master** database is selected, and then run the script to create the **adventureworks** database. This will take a few minutes.
5. After the database has been created, on the **Connections** pane, in the **Servers** section, create a new connection with the following settings:
    - **Connection type**: SQL Server
    - **Server**: localhost\SQLExpress
    - **Authentication Type**: Windows Authentication
    - **Database**: adventureworks
    - **Server group**: \<Default\>
    - **Name**: AdventureWorks

## Challenges
[Retrieve customer data](#retrieve-customer-data)

[Retrieve customer names and phone numbers](#retrieve-customer-names-and-phone-numbers)

[Retrieve customer order data](#retrieve-customer-order-data)

[Retrieve customer contact details](#retrieve-customer-contact-details)

[Retrieve data for transportation reports](#retrieve-data-for-transportation-reports)

[Retrieve product data](#retrieve-product-data)

[Generate invoice reports and return customer information](#generate-invoice-reports-and-return-customer-information)

[Profit Analysis](#profit-analysis)

[Shipping orders and product sales](#shipping-orders-and-product-sales)

[Modifying tables](#modifying-tables)

### Retrieve customer data
_Familiarise yourself with the_ **SalesLT.Customer** _table by writing a Transact-SQL query that retrieves all columns for all customers. Then, create a list of all customer contact names that includes the title, first name, middle name (if any), last name, and suffix (if any) of all customers._

#### _My solution:_
To retrieve all columns for all customers from the SalesLT.Customer table, we can simply do:
```sql
SELECT * FROM SalesLT.Customer
```
Inspecting the DB, it is clear there are multiple ways to create the list of all customer contact names. However, given that middle names and suffixes are optional, I see this as an opportunity to learn the syntax behind the ```CASE``` expression within the SQL language and of any differences from other programming languages.
```sql
SELECT
    CASE
        WHEN Title IS NOT NULL AND MiddleName IS NOT NULL AND Suffix IS NOT NULL THEN
            Title + ' ' + FirstName + ' ' + MiddleName + ' ' + LastName + ' ' + Suffix
        WHEN TITLE IS NOT NULL AND MiddleName IS NOT NULL AND Suffix IS NULL THEN
            Title + ' ' + FirstName + ' ' + MiddleName + ' ' + LastName
        WHEN Title IS NOT NULL AND MiddleName IS NULL AND Suffix IS NOT NULL THEN
            Title + ' ' + FirstName + ' ' + LastName + ' ' + Suffix   
        WHEN Title IS NOT NULL AND MiddleName IS NULL AND Suffix IS NULL THEN
            Title + ' ' + FirstName + ' ' + LastName   
        WHEN Title IS NULL AND MiddleName IS NOT NULL AND Suffix IS NOT NULL THEN
            FirstName + ' ' + MiddleName + ' ' + LastName + ' ' + Suffix
        WHEN TITLE IS NULL AND MiddleName IS NOT NULL AND Suffix IS NULL THEN
            FirstName + ' ' + MiddleName + ' ' + LastName
        WHEN Title IS NULL AND MiddleName IS NULL AND Suffix IS NOT NULL THEN
            FirstName + ' ' + LastName + ' ' + Suffix   
        WHEN Title IS NULL AND MiddleName IS NULL AND Suffix IS NULL THEN
            FirstName + ' ' + LastName            
    END
    AS CustomerNames
FROM SalesLT.Customer
```
### Retrieve customer names and phone numbers
_Each customer has an assigned salesperson. You must write a query to create a call sheet that lists:_
- _the sales person,_
- _a column named_ "CustomerName" _that displays how the customer contact should be greeted (for example,_ "Mr Smith"_),_
- _the customer's phone number._

#### _My solution:_
```sql
SELECT
    SalesPerson,
    CASE
        WHEN Title IS NULL THEN Firstname
        ELSE Title + ' ' + LastName
    END AS CustomerName,
    Phone AS PhoneNumber
FROM SalesLT.Customer
```
### Retrieve customer order data
_Retrieve a list of customer companies which adopts the following format:_ "Customer ID: Company Name" _(for example,_ "78: Preferred Bikes" _). Then, query the_ **SalesLT.SalesOrderHeader** _table to retrieve data for a report that shows:_
- _the sales order number and revision number in the format:_ "SOXXXXX (X)" _(for example,_ SO71774 (2) _),_
- _the order date converted to ANSI standard 102 format:_ yyyy.mm.dd _(for example,_ 2015.01.31 _)._

#### _My solution:_
```sql
SELECT 
    SalesOrderNumber + ' (' + TRY_CONVERT(nvarchar(3), RevisionNumber) +')' AS SalesOrder_Number_Revision,
    CONVERT(nvarchar(30), OrderDate, 102) AS ANSI_Standard_102_Format    
FROM SalesLT.SalesOrderHeader
```
### Retrieve customer contact details
1. _Write a query which retrieves customer contact names (with middle names if known). The list must consist of a single column in the format_ "First Last" _(for example,_ "Keith Harris" _) if the middle name is unknown, or_ "First Middle Last" _(for example,_ "Jane M. Gates" _) if a middle name is known._
2. _Customers may provide Adventure Works with an email address, a phone number, or both. If an email address is available, then it should be used as the primary contact method; if not, then the phone number should be used. You must write a query that returns a list of customer IDs in one column, and a second column named_ "PrimaryContact" _that contains the email address if known, and otherwise the phone number. **IMPORTANT**: In the sample data provided, there are no customer records without an email address. Therefore, to verify that your query works as expected, remove some existing email addresses before creating your query._
3. _You have been asked to create a query that returns a list of sales order IDs and order dates with a column named_ "ShippingStatus" _that contains the text_ 'Shipped' _for orders with a known ship date, and_ 'Awaiting Shipment' _for orders with no ship date. **IMPORTANT**: In the sample data provided, there are no sales order header records without a ship date. Therefore to verify that your query works as expected, remove some existing ship dates before creating your query._

#### _My solution:_
The first component of this challenge is similar to the previous, in that we are retrieving the customer contact names. My attempt at a simple solution is:
```sql
SELECT
    CASE
        WHEN MiddleName IS NULL THEN FirstName + ' ' + LastName
        ELSE FirstName + ' ' + MiddleName + ' ' + LastName
    END AS CustomerContactNames
FROM SalesLT.Customer
```
For the next component, I must first remove some entries within the 'EmailAddress' field of the **SalesLT.Customer** table. As a precautionary measure, I first produce a copy of the table which will be used in my solution.
```sql
SELECT * INTO SalesLT.Customer_copy FROM SalesLT.Customer
```
Then, removing some existing email addresses before creating my query can be done the following way:
```sql
UPDATE SalesLT.Customer_copy
SET EmailAddress = NULL
WHERE CustomerID % 7 = 1;
```
Now, implementing the query can again be performed with a ```CASE``` expression:
```sql
SELECT 
    CustomerID,
    CASE
        WHEN EmailAddress IS NULL THEN Phone
        ELSE EmailAddress
    END AS ContactDetails
FROM SalesLT.Customer_copy
```
As for the final component, the table containing the shipping status is located in the **SalesOrderHeader** table. I again create a copy of the table and remove some existing shipping dates.
```sql
SELECT * INTO SalesLT.SalesOrderHeader_copy FROM SalesLT.SalesOrderHeader
UPDATE SalesLT.SalesOrderHeader_copy
SET ShipDate = NULL
WHERE SalesOrderID > 71899;
```
Finally, this query can also be easily implemented using the following ```CASE``` expression:
```sql
SELECT
    SalesOrderID,
    CONVERT(nvarchar(30), OrderDate, 102) AS OrderDate,
    CASE
        WHEN ShipDate IS NULL THEN 'Awaiting Shipment'
        ELSE 'Shipped'
    END AS ShippingStatus
FROM SalesLT.SalesOrderHeader_copy
```
### Retrieve data for transportation reports
_Produce a list of all customer locations by writing a T-SQL query that retrieves the values for_ City _and_ StateProvince, _removing duplicates and sorted in ascending order of city._

#### _My solution:_
Inspecting the table, we see that the "City" and "StateProvince" values are stored in the **SalesLT.Address** table. I can write the query as:
```sql
SELECT DISTINCT City, StateProvince FROM SalesLT.Address 
ORDER BY City ASC
```
_Transportation costs are increasing and you need to identify the heaviest products. Retrieve the names of the top 10% of products by weight._

The information regarding product weights are contained in the **SalesLT.Product** table, with my solution becoming:
```sql
SELECT TOP (10) PERCENT  WITH TIES Name FROM SalesLT.Product
ORDER BY Weight DESC
```
### Retrieve product data
_Find the names, colours, and sizes of the products with a product model ID 1._

#### _My solution:_
The following solution is implemented with the ```WHERE``` clause, which serves as a condition for the preceding ```SELECT``` statement.
```sql
SELECT ProductModelID, Name, Color, Size FROM SalesLT.Product
WHERE ProductModelID = 1
```
_Retrieve the product number and name of the products that have a colour of black, red or white, and a size of S or M._

#### _My solution:_
```sql
SELECT ProductNumber, Name, Color, Size FROM SalesLT.Product
WHERE Color IN ('Black','Red','White') AND Size IN ('S','M')
ORDER BY Name
```
_Retrieve the product number, name, and list price of products whose product number begins_ 'BK-'_. Then, modify the query to retrieve the product number, name, and list price of products whose product number begins_ 'BK-' _followed by any character other than_ 'R'_, and ends with a_ '-' _followed by any two numerals._

#### _My solution:_
```sql
SELECT ProductNumber, Name, ListPrice FROM SalesLT.Product
WHERE ProductNumber LIKE 'BK-[^R][0-9][0-9][A-Z]-[0-9][0-9]'
```
### Generate invoice reports and return customer information
_As an initial step towards generating the invoice report, write a query that returns the company name from the_ **SalesLT.Customer** _table, and the sales order ID and total due from the_ SalesLT.SalesOrderHeader _table. Then, extend your customer orders query to include the Main Office address for each customer, including the full street address, city, state or province, postal code, and country or region._

#### _My solution:_
This challenge entails the use of joins as I will be needing to query multiple tables using a primary key. This is especially so as each customer can have multiple addressees in the **SalesLT.Address** table, with the **SalesLT.CustomerAddress** table enabling a many-to-many relationship between customers and addresses.

To address the first component of the challenge, we can join the **SalesLT.Customer** and **SalesLT.SalesOrderHeader** like so:
```sql
SELECT c.CompanyName, oh.SalesOrderID, oh.TotalDue
FROM SalesLT.Customer AS c
INNER JOIN SalesLT.SalesOrderHeader AS oh
    ON c.CustomerID = oh.CustomerID
```

Now, to address the second component of the challenge, I need to include both the **SalesLT.Address** and **SalesLT.CustomerAddress** tables. Considering the headers I am required to include for the invoice report, I can do this by constructing the following join:
```sql
SELECT c.CompanyName, oh.SalesOrderID, oh.TotalDue, ca.AddressType, a.AddressLine1, a.City, a.StateProvince, a.PostalCode, a.CountryRegion
FROM SalesLT.Customer AS c
INNER JOIN SalesLT.SalesOrderHeader AS oh
    ON c.CustomerID = oh.CustomerID
INNER JOIN SalesLT.CustomerAddress AS ca
    ON oh.CustomerID = ca.CustomerID
INNER JOIN SalesLT.Address as a
    ON ca.AddressID = a.AddressID
WHERE ca.AddressType = 'Main Office'
```
Note that the final line, ```WHERE ca.AddressType = 'Main Office'```, filters the results so that only Main Office addresses are included.

_The sales manager wants a list of all customer companies and their contacts (first name and last name), showing the sales order ID and total due for each order they have placed. Customers who have not placed any orders should be included at the bottom of the list with NULL values for the order ID and total due._

#### _My solution:_
```sql
SELECT c.CompanyName, c.FirstName, c.LastName, oh.SalesOrderID, oh.TotalDue
FROM SalesLT.Customer AS c
LEFT JOIN SalesLT.SalesOrderHeader as oh
    ON c.CustomerID = oh.CustomerID
ORDER BY oh.SalesOrderID DESC
```

_A sales employee has noticed that Adventure Works does not have address information for all customers. You must write a query that returns a list of customer IDs, company names, contact names (first name and last name), and phone numbers for customers with no address stored in the database._

#### _My solution:_
```sql
SELECT c.CustomerID, c.CompanyName, c.FirstName, c.LastName, c.Phone
FROM SalesLT.Customer as c
LEFT OUTER JOIN SalesLT.CustomerAddress as ca
    ON c.CustomerID = ca.CustomerID
WHERE ca.AddressID IS NULL
```
_The product catalogue will list products by parent category and subcategory, so you must write a query that retrieves the parent category name, subcategory name, and product name fields for the catalogue._

#### _My solution:_
```sql
SELECT pcat.Name AS ParentCategory, scat.Name AS SubCategory, p.Name AS ProductName
FROM SalesLT.Product as p
JOIN SalesLT.ProductCategory as scat
    ON p.ProductCategoryID = scat.ProductCategoryID
JOIN SalesLT.ProductCategory as pcat
    ON scat.ParentProductCategoryID = pcat.ProductCategoryID
ORDER BY ParentCategory, SubCategory, ProductName
```
### Profit Analysis
_Retrieve the product ID, name and list price for each product where the list price is higher than the average unit price for all products that have been sold._

#### _My solution:_
To retrieve the average value, I can use the ```AVG``` function, while using the ```WHERE``` clause to seek those products whose list prices are greater than the average unit price. However, this ```WHERE``` clause must be structured correctly in order to properly embed the accompanying ```SELECT``` clause. My solution is as follows:
```sql
SELECT ProductID, Name, ListPrice
FROM SalesLT.Product
WHERE ListPrice >
    (SELECT AVG(od.UnitPrice)
    FROM SalesLT.SalesOrderDetail AS od)
ORDER BY ProductID
```

_Retrieve the product ID, name and list price for each product where the list price is 100 or more, and the product has been sold for less than 100._

#### _My solution:_
This challenge was interesting to me for two reasons. Firstly, due to how the unit price for each product could be interpreted, and secondly, due to how the solution could be implemented using either a subquery or join. Inspecting the **SalesLT.SalesOrderDetail** table, I could see that the same product had different unit prices depending on the order quantity. For example, considering rows 14 and 15 of the **SalesLT.SalesOrderDetail** table:
```
+--------------+----------+-----------+-----------+-------------------+
| SalesOrderID | OrderQty | ProductID | UnitPrice | UnitPriceDiscount |
+--------------+----------+-----------+-----------+-------------------+
|    71782     |    6     |    711    |   20.994  |       0.00        | 
+--------------+----------+-----------+-----------+-------------------+
|    71783     |    15    |    711    |  19.2445  |       0.05        |
+--------------+----------+-----------+-----------+-------------------+
```

I can see that for some product with ProductID = 711, its unit price is either 20.994 or 19.2445. Clearly, this is due to a discount taking effect when the order quantity surpasses a specific value. Then at this point, it is unclear whether the challenge intends for us to consider the unit price of 100 to be the discounted or non-discounted value. Out of interest, I developed the following query to return the maximum (i.e. non-discounted) unit price for each product:
```sql
-- Return the maximum (i.e. non-discounted) unit price for each product
SELECT DISTINCT ProductID, UnitPrice
FROM SalesLT.SalesOrderDetail AS od1
WHERE UnitPrice = 
    (SELECT MAX(UnitPrice)
    FROM SalesLT.SalesOrderDetail AS od2
    WHERE od2.ProductID = od1.ProductID)
```
After listing this result, I decided to forgo any ambiguities regarding the unit price by simply considering the discounted value. Then, it was a matter of framing my solution using either a subquery or join. For the former case, my solution was structured using the following subquery:
```sql
-- Solution using subqueries
SELECT ProductID, Name, ListPrice
FROM SalesLT.Product
WHERE ProductID IN
    (SELECT DISTINCT ProductID
    FROM SalesLT.SalesOrderDetail
    WHERE UnitPrice < 100)
AND (ListPrice >= 100)
```
For the latter case, my solution implementing joins becomes:
```sql
-- Solution using joins
SELECT DISTINCT p.ProductID, p.Name, p.ListPrice, od.UnitPrice
FROM SalesLT.Product AS p
INNER JOIN SalesLT.SalesOrderDetail AS od
    ON p.ProductID = od.ProductID
WHERE (p.ListPrice >= 100) AND (od.UnitPrice < 100)
ORDER BY p.ProductID
```
_Retrieve the product ID, name, cost and list price for each product along with the average unit price for which that product has been sold._

#### _My solution:_
```sql
SELECT p.ProductID, p.Name, p.StandardCost AS Cost, p.ListPrice,
    (SELECT AVG(UnitPrice)
        FROM SalesLT.SalesOrderDetail AS od
        WHERE p.ProductID = od.ProductID) AS AvgUnitPrice
FROM SalesLT.Product AS p
ORDER BY p.ProductID
```
_Filter your previous query to include only products where the cost price is higher than the average selling price._

#### _My solution:_
```sql
SELECT p.ProductID, p.Name, p.StandardCost AS Cost, p.ListPrice,
    (SELECT AVG(UnitPrice)
        FROM SalesLT.SalesOrderDetail AS od
        WHERE od.ProductID = p.ProductID) AS AvgUnitPrice
FROM SalesLT.Product AS p
WHERE p.StandardCost > 
        (SELECT AVG(UnitPrice)
        FROM SalesLT.SalesOrderDetail AS od
        WHERE od.ProductID = p.ProductID)
ORDER BY p.ProductID
```
### Shipping orders and product sales

- _Write a query to return the order ID for each order, together with the_ Freight _value rounded to two decimal places in a column named_ "FreightCost".
- _Extend your query to include a column named_ "ShippingMethod" _that contains the_ ShipMethod _field, formatted in lower case._
- _Extend your query to include columns named_ "ShipYear", "ShipMonth" _and_ "ShipDay" _that contain the year, month and day of the_ ShipDate. _The_ ShipMonth _value should be displayed as the month name (for example,_ "June"_)._

#### _My solution:_

For the first challenge component, I can formulate the query as:
```sql
SELECT SalesOrderID, ROUND(Freight, 2) AS FreightCost
FROM SalesLT.SalesOrderHeader
ORDER BY SalesOrderID
```
Then for the second component, the above query can be extended to include those desired fields:
```sql
SELECT SalesOrderID, ROUND(Freight, 2) AS FreightCost, LOWER(ShipMethod) AS ShippingMethod
FROM SalesLT.SalesOrderHeader
ORDER BY SalesOrderID
```
Finally, utilising the ```DATENAME``` function, the query for last component can be written as:
```sql
SELECT SalesOrderID, 
    ROUND(Freight, 2) AS FreightCost, 
    LOWER(ShipMethod) AS ShippingMethod,
    DATENAME(yy, ShipDate) AS ShipYear,
    DATENAME(mm, ShipDate) AS ShipMonth,
    DATENAME(dw, ShipDate) AS ShipDay
FROM SalesLT.SalesOrderHeader
ORDER BY SalesOrderID
```
- _Write a query to retrieve a list of the product names from the_ **SalesLT.Product** _table and the total revenue for each product calculated as the sum of_ LineTotal _from the_ **SalesLT.SalesOrderDetail** _table, with the results sorted in descending order of total revenue._
- _Modify the previous query to only include sales totals for products that have a list price of more than 1000._
- _Modify the previous query to only include the product groups with a total sales value greater than 20000._

#### _My solution:_

For this challenge, the first component invites the use of the join clause using the primary key linking the **SalesLT.Product** and **SalesLT.SalesOrderDetail** tables. This can be done like so:
```sql
SELECT p.Name, SUM(od.LineTotal) AS TotalRevenue
FROM SalesLT.Product AS p
JOIN SalesLT.SalesOrderDetail AS od
    ON p.ProductID = od.ProductID
GROUP BY p.Name
ORDER BY TotalRevenue DESC
```
To solve the next component, I modify the previous query using a ```WHERE``` clause to fit the criteria for those products whose sales totals are greater than 1000.
```sql
SELECT p.Name, SUM(od.LineTotal) AS TotalRevenue
FROM SalesLT.Product AS p
JOIN SalesLT.SalesOrderDetail AS od
    ON p.ProductID = od.ProductID
WHERE p.ListPrice > 1000
GROUP BY p.Name
ORDER BY TotalRevenue DESC
```
Then for the final component, the last query can be easily modified with the ```HAVING``` clause:
```sql
SELECT p.Name, SUM(od.LineTotal) AS TotalRevenue
FROM SalesLT.Product AS p
JOIN SalesLT.SalesOrderDetail AS od
    ON p.ProductID = od.ProductID
WHERE p.ListPrice > 1000
GROUP BY p.Name
HAVING SUM(od.LineTotal) > 20000
ORDER BY TotalRevenue DESC
```
### Modifying tables
_Adventure Works has started selling the following new product. Insert it into the_ **SalesLT.Product** _table, using default or_ NULL _values for unspecified columns:_
- _Name:_ LED Lights
- _ProductNumber:_ LT-L123
- _StandardCost:_ 2.56
- _ListPrice:_ 12.99
- _ProductCategoryID:_ 37
- _SellStartDate:_ Today's date

#### _My solution:_
```sql
INSERT INTO SalesLT.Product (Name, ProductNumber, StandardCost, ListPrice, ProductCategoryID, SellStartDate)
VALUES
('LED Lights', 'LT-L123', 2.56, 12.99, 37, GETDATE())
```
_After you have inserted the product, run a query to determine the ProductID that was generated. Then, run a query to view the row for the product in the_ **SalesLT.Product** _table._
#### _My solution:_
```sql
SELECT SCOPE_IDENTITY()

SELECT * FROM SalesLT.Product
WHERE ProductID = SCOPE_IDENTITY()
```

_Adventure Works is adding a product category for 'Bells and Horns' to its catalogue. The parent category for the new category is_ 4 (Accessories). _This new category includes the following two new products:_
- _First product:_
    - _Name:_ Bicycle Bell
    - _ProductNumber:_ BB-RING
    - _StandardCost:_ 2.47
    - _ListPrice:_ 4.99
    - _ProductCategoryID:_ The ProductCategoryID for the new Bells and Horns category
    - _SellStartDate:_ Today's date
- _Second product:_
    - _Name:_ Bicycle Horn
    - _ProductNumber:_ BB-PARP
    - _StandardCost:_ 1.29
    - _ListPrice:_ 3.75
    - _ProductCategoryID:_ The ProductCategoryID for the new Bells and Horns category
    - _SellStartDate:_ Today's date

#### _My solution:_

I begin by writing a query to insert the new product category:
```sql
INSERT INTO SalesLT.ProductCategory (ParentProductCategoryID, Name)
VALUES
(4, 'Bells and Horns')
```

Then, inserting the two new products with the appropriate ProductCategoryID value, I structure the query as:
```sql
SELECT IDENT_CURRENT('SalesLT.ProductCategory') AS LatestProductCategoryID
INSERT INTO SalesLT.Product (Name, ProductNumber, StandardCost, ListPrice, ProductCategoryID, SellStartDate)
VALUES
('Bicycle Bell', 'BB-RING', 2.47, 4.99, LatestProductCategoryID, GETDATE()),
('Bicycle Horn', 'BB-RING', 1.29, 3.75, LatestProductCategoryID, GETDATE())
```


