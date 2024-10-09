-- CHALLENGE 1: Retrieve order shipping information
-- 1. Retrieve the order ID and freight cost of each order
--    Write a query to return the order ID for each order, together with the Freight value rounded to two decimal places in a column named FreightCost.

SELECT SalesOrderID, ROUND(Freight, 2) AS FreightCost
FROM SalesLT.SalesOrderHeader
ORDER BY SalesOrderID

-- 2. Add the shipping method
--    Extend your query to include a column named ShippingMethod that contains the ShipMethod field, formatted in lower case.

SELECT SalesOrderID, ROUND(Freight, 2) AS FreightCost, LOWER(ShipMethod) AS ShippingMethod
FROM SalesLT.SalesOrderHeader
ORDER BY SalesOrderID

-- 3. Add shipping date details
--    Extend your query to include columns named ShipYear, ShipMonth and ShipDay that contain the year, month and day of the ShipDate. The ShipMonth value should be displayed as the month name (for example, June).

SELECT SalesOrderID, 
    ROUND(Freight, 2) AS FreightCost, 
    LOWER(ShipMethod) AS ShippingMethod,
    DATENAME(yy, ShipDate) AS ShipYear,
    DATENAME(mm, ShipDate) AS ShipMonth,
    DATENAME(dw, ShipDate) AS ShipDay
FROM SalesLT.SalesOrderHeader
ORDER BY SalesOrderID

-- CHALLENGE 2: Aggregate product sales
-- 1. Retrieve total sales by product
--    Write a query to retrieve a list of the product names from the SalesLT.Product table and the total revenue for each product calculated as the sum of LineTotal from the SalesLT.SalesOrderDetail table, with the results sorted in descending order of total revenue.

SELECT p.Name, SUM(od.LineTotal) AS TotalRevenue
FROM SalesLT.Product AS p
JOIN SalesLT.SalesOrderDetail AS od
    ON p.ProductID = od.ProductID
GROUP BY p.Name
ORDER BY TotalRevenue DESC

-- 2. Filter the product sales list to include only products that cost over 1000.
--    Modify the previous query to only include sales totals for products that have a list price of more than 1000.

SELECT p.Name, SUM(od.LineTotal) AS TotalRevenue
FROM SalesLT.Product AS p
JOIN SalesLT.SalesOrderDetail AS od
    ON p.ProductID = od.ProductID
WHERE p.ListPrice > 1000
GROUP BY p.Name
ORDER BY TotalRevenue DESC

-- 3. Filter the product sales groups to include only total sales over 20000.
--    Modify the previous query to only include the product groups with a total sales value greater than 20000.

SELECT p.Name, SUM(od.LineTotal) AS TotalRevenue
FROM SalesLT.Product AS p
JOIN SalesLT.SalesOrderDetail AS od
    ON p.ProductID = od.ProductID
WHERE p.ListPrice > 1000
GROUP BY p.Name
HAVING SUM(od.LineTotal) > 20000
ORDER BY TotalRevenue DESC