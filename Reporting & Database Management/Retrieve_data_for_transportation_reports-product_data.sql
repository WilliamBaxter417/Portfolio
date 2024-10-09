-- CHALLENGE 1: Retrieve data for transportation reports
-- 1. Retrieve a list of cities
--    Initially, you need to produce a list of all customer locations. Write a T-SQL query that queries the SalesLT.Address table and retrieves the values for City and StateProvince, removing duplicates and sorted in ascending order of city.

SELECT DISTINCT City, StateProvince FROM SalesLT.Address 
ORDER BY City ASC

-- 2. Retrieve the heaviest products
--    Transportation costs are increasing and you need to identify the heaviest products. Retrieve the names of the top 10% of products by weight.

SELECT TOP (10) PERCENT  WITH TIES Name FROM SalesLT.Product
ORDER BY Weight DESC

-- CHALLENGE 2: Retrieve product data
-- 1. Retrieve product details for product model 1
--    Initially, you need to find the names, colours, and sizes of the products with a product model ID 1.

SELECT ProductModelID, Name, Color, Size FROM SalesLT.Product
WHERE ProductModelID = 1

-- 2. Filter products by colour and size
--   Retrieve the product number and name of the products that have a colour of black, red or white, and a size of S or M.

SELECT ProductNumber, Name, Color, Size FROM SalesLT.Product
WHERE Color IN ('Black','Red','White') AND Size IN ('S','M')
ORDER BY Name

-- 3. Filter products by product number
--    Retrieve the product number, name, and list price of products whose product number begins 'BK-'

SELECT ProductNumber, Name, ListPrice FROM SalesLT.Product
WHERE ProductNumber LIKE 'BK-%'

-- 4. Retrieve specific products by product number
--    Modify your previous query to retrieve the product number, name, and list price of products whose product number begins 'BK-' followed by any character other than 'R', and ends with a '-' followed by any two numerals.

SELECT ProductNumber, Name, ListPrice FROM SalesLT.Product
WHERE ProductNumber LIKE 'BK-[^R][0-9][0-9][A-Z]-[0-9][0-9]'
-- WHERE ProductNumber LIKE 'BK-[^R]%-[0-9][0-9]'  (an alternative)

-- SELECT TOP (10) * FROM SalesLT.Address