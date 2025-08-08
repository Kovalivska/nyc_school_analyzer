# 🧾 NYC High School Directory – Dataset Structure

Below is a description of the key columns found in the high school directory dataset. This reference will help you understand what each field represents as you clean, explore, and analyze the data.

| Column Name                      | Description                                                                                      |
|----------------------------------|--------------------------------------------------------------------------------------------------|
| DBN                              | District Borough Number, a unique code identifying each school (e.g., 01M364)                   |
| Boro                             | Borough Code                                                                                     |
| BN                               | Borough Number                                                                                   |
| Building Code                    | Code for the building the school is located in (not always same as DBN)                          |
| School_Phone_Number              | Contact number for the school                                                                    |
| Fax_Number                       | Fax number                                                                                       |
| Printed_Name                     | School's printed display name                                                                    |
| Alphabetic_Name_Long             | Alternate formatting of name (e.g., "Facing History School, The")                               |
| grade span 2014-2015 min         | Minimum grade offered during 2014-2015 (e.g., 0K = Kindergarten)                                |
| grade span 2014-2015 max         | Maximum grade offered during 2014-2015                                                           |
| Expected grade span min          | Expected minimum grade                                                                           |
| Expected grade span max          | Expected maximum grade                                                                           |
| Last Year BUS?                   | If 1, bus routes are from the prior year's analysis                                              |
| BUS                              | Available bus routes                                                                             |
| Last Year SUBWAY?                | If 1, subway access is from prior year's analysis                                                |
| SUBWAY                           | Available subway lines                                                                           |
| Primary_Address_Line_1           | Street address                                                                                   |
| City                             | City or borough (e.g., "New York" for Manhattan)                                                 |
| State_Code                       | NY (New York)                                                                                    |
| Zip                              | ZIP Code                                                                                         |
| Total Student 10/26              | Number of students enrolled as of October 26                                                     |
| Campus_Name                      | Name of the campus                                                                               |
| ELL Data                         | English Language Learners data                                                                   |
| School_Accessibility_Description | Indicates if the school is fully, partially, or not accessible                                  |
| First Priority01 – Tenth Priority10 | Admission priorities by rank                                                                 |
| progcount                        | Number of programs offered                                                                       |
| Email                            | School's email contact                                                                           |
| Independent Website              | Official website URL                                                                             |
| School_Type                      | Type of institution (e.g., public, charter)                                                      |
| Overview Paragraph               | Summary description of the school                                                                |
| Program Highlights               | Notable features or offerings                                                                    |
| Language Classes                 | Foreign languages offered                                                                        |
| Advanced Placement Courses       | AP classes available                                                                             |
| PSAL Sports - Boys/Girls/Co-ed   | School sports offered by gender                                                                  |
| School Sports                    | General list of school sports                                                                    |
| Start Time / End Time            | School's operating hours                                                                         |
| SE_Services                      | Special education services                                                                       |
| Online AP/Language Courses       | Online courses availability                                                                      |
| Neighborhood, Districts, Latitude, Longitude | External geographic fields from Ontodia                                             |
| @context / @type / @id           | Metadata fields (Ontodia context)                                                                |

---

📌 Use this table as a reference when analyzing or cleaning the dataset in Day 2’s Python task.
Link to original data from Kaggle: [https://www.kaggle.com/datasets/new-york-city/nyc-high-school-directory?select=doe-high-school-directory-2013-2014.csv]
