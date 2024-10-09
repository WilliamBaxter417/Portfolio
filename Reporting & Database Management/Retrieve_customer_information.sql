-- CHALLENGE 1: Retrieve customer data
-- 1. Retrieve customer details
--    Familiarise yourself with the SalesLT.Customer table by writing a Transact-SQL query that retrieves all columns for all customers.
SELECT * FROM SalesLT.Customer

-- 2. Retrieve customer name data
--    Create a list of all customer contact names that includes the title, first name, middle name (if any), last name, and suffix (if any) of all customers.

-- INITIAL SOLUTION: Implemented with what is called a 'searched case'. This differs from a 'simple case' whereby a column header is not referenced immediately following the 'CASE' expression.
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

-- 3. Retrieve customer names and phone numbers
--    Each customer has an assigned salesperson. You must write a query to create a call sheet that lists:
--      - the sales person,
--      - a column named CustomerName that displays how the customer contact should be greeted (for example, Mr Smith),
--      - the customer's phone number.
SELECT
    SalesPerson,
    CASE
        WHEN Title IS NULL THEN Firstname
        ELSE Title + ' ' + LastName
    END AS CustomerName,
    Phone AS PhoneNumber
FROM SalesLT.Customer

-- CHALLENGE 2: Retrieve customer order data
-- 1. Retrieve a list of customer companies.
--    You have been asked to provide a list of all customer companies in the format:
--    'Customer ID: Company Name' (for example, '78: Preferred Bikes')
SELECT 
    CASE 
        WHEN CompanyName IS NULL THEN 
            CONVERT(nvarchar(5), CustomerID) + ': ' + 'Company Name N/A'
        ELSE CONVERT(nvarchar(5), CustomerID) + ': ' + CompanyName
    END AS CustomerCompanyList
FROM SalesLT.Customer

-- 2. Retrieve a list of sales order revisions
--    The SalesLT.SalesOrderHeader table contains records of sales orders. You have been asked to retrieve data for a report that shows:
--     - The sales order number and revision number in the format 'SOXXXXX (X)' - for example, SO71774 (2).
--     - The order date converted to ANSI standard 102 format (yyyy.mm.dd - for example 2015.01.31)
SELECT 
    SalesOrderNumber + ' (' + TRY_CONVERT(nvarchar(3), RevisionNumber) +')' AS SalesOrder_Number_Revision,
    CONVERT(nvarchar(30), OrderDate, 102) AS ANSI_Standard_102_Format    
FROM SalesLT.SalesOrderHeader

-- CHALLENGE 3: Retrieve customer contact details
-- 1. Retrieve customer contact names with middle names if known.
--    You have been asked to write a query that returns a list of customer names. The list must consist of a single column in the format 'first last' (for example, 'Keith Harris') if the middle is unknown, or 'first middle last' (for example, 'Jane M. Gates') if a middle name is known.
SELECT
    CASE
        WHEN MiddleName IS NULL THEN FirstName + ' ' + LastName
        ELSE FirstName + ' ' + MiddleName + ' ' + LastName
    END AS CustomerContactNames
FROM SalesLT.Customer

-- 2. Retrieve primary contact details
--    Customers may provide Adventure Works with an email address, a phone number, or both. If an email
--    address is available, then it should be used as the primary contact method; if not, then the phone
--    number should be used. You must write a query that returns a list of customer IDs in one column,
--    and a second column named 'PrimaryContact' that contains the email address if known, and otherwise
--    the phone number.
--    IMPORTANT: In the sample data provided, there are no customer records without an email address.
--    Therefore, to verify that your query works as expected, run the following UPDATE statement to remove
--    some existing email addresses before creating your query.

-- Make a copy of the existing Customer table before removing some email addresses
SELECT * INTO SalesLT.Customer_copy FROM SalesLT.Customer
-- Remove some existing email addresses before creating query
UPDATE SalesLT.Customer_copy
SET EmailAddress = NULL
WHERE CustomerID % 7 = 1;
-- Begin query
SELECT 
    CustomerID,
    CASE
        WHEN EmailAddress IS NULL THEN Phone
        ELSE EmailAddress
    END AS ContactDetails
FROM SalesLT.Customer_copy

-- 3. Retrieve shipping status
--    You have been asked to create a query that returns a list of sales order IDs and order dates with
--    a column named ShippingStatus that contains the text 'Shipped' for orders with a known ship date,
--    and 'Awaiting Shipment' for orders with no ship date.
--    IMPORTANT: In the sample data provided, there are no sales order header records without a ship date.
--    Therefore to verify that your query works as expected, run the following UPDATE statement to remove
--    some existing ship dates before creating your query.

-- Make a copy of the existing SalesOrderHeader table before removing some ship dates
SELECT * INTO SalesLT.SalesOrderHeader_copy FROM SalesLT.SalesOrderHeader
-- Remove some existing ship dates before creating query
UPDATE SalesLT.SalesOrderHeader_copy
SET ShipDate = NULL
WHERE SalesOrderID > 71899;
-- Begin query
SELECT
    SalesOrderID,
    CONVERT(nvarchar(30), OrderDate, 102) AS OrderDate,
    CASE
        WHEN ShipDate IS NULL THEN 'Awaiting Shipment'
        ELSE 'Shipped'
    END AS ShippingStatus
FROM SalesLT.SalesOrderHeader_copy




--SELECT TOP(100) * FROM SalesLT.Customer_copy