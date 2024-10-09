-- CHALLENGE 1: Retrieve product price information
-- 1. Retrieve products whose list price is higher than the average unit price.
--    Retrieve the product ID, name and list price for each product where the list price is higher than the average unit price for all products that have been sold.
--    TIP: Use the AVG function to retrieve an average value.

SELECT ProductID, Name, ListPrice
FROM SalesLT.Product
WHERE ListPrice >
    (SELECT AVG(od.UnitPrice)
    FROM SalesLT.SalesOrderDetail AS od)
ORDER BY ProductID

-- 2. Retrieve products with a list price of 100 or more that have been sold for less than 100.
--    Retrieve the product ID, name and list price for each product where the list price is 100 or more, and the product has been sold for less than 100.

-- THIS SOLUTION IMPLEMENTS A JOIN (NOT DESIRED BY QUESTION)
/*SELECT DISTINCT p.ProductID, p.Name, p.ListPrice, od.UnitPrice
FROM SalesLT.Product AS p
INNER JOIN SalesLT.SalesOrderDetail AS od
    ON p.ProductID = od.ProductID
WHERE (p.ListPrice >= 100) AND (od.UnitPrice < 100)
ORDER BY p.ProductID*/

-- THIS SOLUTION IMPLEMENTS A SUBQUERY (AS DESIRED BY QUESTION)
SELECT ProductID, Name, ListPrice
FROM SalesLT.Product
WHERE ProductID IN
    (SELECT DISTINCT ProductID
    FROM SalesLT.SalesOrderDetail
    WHERE UnitPrice < 100)
AND (ListPrice >= 100)

-- THIS CODE RETURNS THE MAXIMUM (i.e. NON-DISCOUNTED) UNIT PRICE FOR EACH PRODUCT
/*SELECT DISTINCT ProductID, UnitPrice
FROM SalesLT.SalesOrderDetail AS od1
WHERE UnitPrice = 
    (SELECT MAX(UnitPrice)
    FROM SalesLT.SalesOrderDetail AS od2
    WHERE od2.ProductID = od1.ProductID)*/

-- CHALLENGE 2: Analyze profitability
-- 1. Retrieve the cost, list price and average selling price for each product.
--    Retrieve the product ID, name, cost and list price for each product along with the average unit price for which that product has been sold.

SELECT p.ProductID, p.Name, p.StandardCost AS Cost, p.ListPrice,
    (SELECT AVG(UnitPrice)
        FROM SalesLT.SalesOrderDetail AS od
        WHERE p.ProductID = od.ProductID) AS AvgUnitPrice
FROM SalesLT.Product AS p
ORDER BY p.ProductID

-- 2. Retrieve products that have an average selling price that is lower than the cost.
--    Filter your previous query to include only products where the cost price is higher than the average selling price.

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
