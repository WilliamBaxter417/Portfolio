-- CHALLENGE 1: Insert products
-- 1. Insert a product
--    Adventure Works has started selling the following new product. Insert it into the SalesLT.Product table, using default or NULL values for unspecified columns:
-- Name: LED Lights
-- ProductNumber: LT-L123
-- StandardCost: 2.56
-- ListPrice: 12.99
-- ProductCategoryID: 37
-- SellStartDate: Today's date
INSERT INTO SalesLT.Product (Name, ProductNumber, StandardCost, ListPrice, ProductCategoryID, SellStartDate)
VALUES
('LED Lights', 'LT-L123', 2.56, 12.99, 37, GETDATE())

-- After you have inserted the product, run a query to determine the ProductID that was generated.
SELECT SCOPE_IDENTITY()

-- Then run a query to view the row for the product in the SalesLT.Product table.
SELECT * FROM SalesLT.Product
WHERE ProductID = SCOPE_IDENTITY()

-- 2. Insert a new category with two products
--    Adventure Works is adding a product category for 'Bells and Horns' to its catalogue. The parent category for the new category is 4 (Accessories). This new category includes the following two new products:
-- First product:
--  Name: Bicycle Bell
--  ProductNumber: BB-RING
--  StandardCost: 2.47
--  ListPrice: 4.99
--  ProductCategoryID: The ProductCategoryID for the new Bells and Horns category
--  SellStartDate: Today's date
-- Second product:
--  Name: Bicycle Horn
--  ProductNumber: BB-PARP
--  StandardCost: 1.29
--  ListPrice: 3.75
--  ProductCategoryID: The ProductCategoryID for the new Bells and Horns category
--  SellStartDate: Today's date

--  Write a query to insert the new product category.
INSERT INTO SalesLT.ProductCategory (ParentProductCategoryID, Name)
VALUES
(4, 'Bells and Horns')

-- Insert the two new products with the appropriate ProductCategoryID value.
SELECT IDENT_CURRENT('SalesLT.ProductCategory') AS LatestProductCategoryID
INSERT INTO SalesLT.Product (Name, ProductNumber, StandardCost, ListPrice, ProductCategoryID, SellStartDate)
VALUES
('Bicycle Bell', 'BB-RING', 2.47, 4.99, LatestProductCategoryID, GETDATE()),
('Bicycle Horn', 'BB-RING', 1.29, 3.75, LatestProductCategoryID, GETDATE())

