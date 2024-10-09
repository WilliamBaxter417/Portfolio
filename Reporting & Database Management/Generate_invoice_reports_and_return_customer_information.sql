-- CHALLENGE 1: Generate invoice reports
-- 1. Retrieve customer orders
--    As an initial step towards generating the invoice report, write a query that returns the company name from the SalesLT.Customer table, and the sales order ID and total due from the SalesLT.SalesOrderHeader table.

SELECT c.CompanyName, oh.SalesOrderID, oh.TotalDue
FROM SalesLT.Customer AS c
INNER JOIN SalesLT.SalesOrderHeader AS oh
    ON c.CustomerID = oh.CustomerID

-- 2. Retrieve customer orders with addresses
--    Extend your customer orders query to include the Main Office address for each customer, including the full street address, city, state or province, postal code, and country or region.
--    TIP: Note that each customer can have multiple addressees in the SalesLT.Address table, so the database developer has created the SalesLT.CustomerAddress table to enable a many-to-many relationship between customers and addresses. Your query will need to include both of these tables, and should filter the results so that only Main Office addresses are included.

SELECT c.CompanyName, oh.SalesOrderID, oh.TotalDue, ca.AddressType, a.AddressLine1, a.City, a.StateProvince, a.PostalCode, a.CountryRegion
FROM SalesLT.Customer AS c
INNER JOIN SalesLT.SalesOrderHeader AS oh
    ON c.CustomerID = oh.CustomerID
INNER JOIN SalesLT.CustomerAddress AS ca
    ON oh.CustomerID = ca.CustomerID
INNER JOIN SalesLT.Address as a
    ON ca.AddressID = a.AddressID
WHERE ca.AddressType = 'Main Office'

-- CHALLENGE 2: Retrieve customer data
-- 1. Retrieve a list of all customers and their orders
--    The sales manager wants a list of all customer companies and their contacts (first name and last name), showing the sales order ID and total due for each order they have placed. Customers who have not placed any orders should be included at the bottom of the list with NULL values for the order ID and total due.

SELECT c.CompanyName, c.FirstName, c.LastName, oh.SalesOrderID, oh.TotalDue
FROM SalesLT.Customer AS c
LEFT JOIN SalesLT.SalesOrderHeader as oh
    ON c.CustomerID = oh.CustomerID
ORDER BY oh.SalesOrderID DESC

-- 2. Retrieve a list of customers with no address
--    A sales employee has noticed that Adventure Works does not have address information for all customers. You must write a query that returns a list of customer IDs, company names, contact names (first name and last name), and phone numbers for customers with no address stored in the database.

SELECT c.CustomerID, c.CompanyName, c.FirstName, c.LastName, c.Phone
FROM SalesLT.Customer as c
LEFT OUTER JOIN SalesLT.CustomerAddress as ca
    ON c.CustomerID = ca.CustomerID
WHERE ca.AddressID IS NULL

-- CHALLENGE 3: Create a product catalogue
-- 1. Retrieve product information by category
--    The product catalogue will list products by parent category and subcategory, so you must write a query that retrieves the parent category name, subcategory name, and product name fields for the catalogue.

SELECT pcat.Name AS ParentCategory, scat.Name AS SubCategory, p.Name AS ProductName
FROM SalesLT.Product as p
JOIN SalesLT.ProductCategory as scat
    ON p.ProductCategoryID = scat.ProductCategoryID
JOIN SalesLT.ProductCategory as pcat
    ON scat.ParentProductCategoryID = pcat.ProductCategoryID
ORDER BY ParentCategory, SubCategory, ProductName