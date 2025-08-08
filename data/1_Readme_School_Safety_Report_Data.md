# 📘 About the School Safety Report Data Dictionary

The file `School_Safety_Report_Data_Dictionary.xlsx` provides essential reference information for working with the **NYC School Safety Report** dataset.

Inside this data dictionary, you'll find:

### 🗂 Dataset Info
General metadata about the dataset, including its purpose, source, and update frequency.

### 📑 Column Info
A structured breakdown of each column in the dataset:
- Column names
- Descriptions and definitions
- Any important nuances or data type hints

### 🕘 Revision History
Details about changes made to the dataset structure or naming over time (if applicable).

---

📌 Use this file as a reference when exploring, cleaning, or uploading the dataset into a database.  
It ensures you understand each field and how it contributes to the larger data structure.

### 📂 About the `school-safety-repor.csv` file

The file `school-safety-report.csv` contains detailed safety report data for various schools across different school years. This dataset can be used to analyze safety trends, compare incident rates between schools, and investigate the geographic distribution of incidents.

#### 📊 Key Columns

* **`Location Name` and `DBN`**: Unique identifiers for each school.
* **`School Year`**: The academic year the data corresponds to.
* **`Major N`, `Oth N`, `NoCrim N`, `Prop N`, `Vio N`**: The count of different types of incidents reported at the school. These can be used to understand the frequency of specific incident categories.
* **`AvgOfMajor N`, `AvgOfOth N`, etc.**: Normalized values representing the average number of incidents, which can be useful for comparing schools of different sizes.
* **`Register`**: The total number of students registered at the school.
* **`Borough Name`, `Latitude`, `Longitude`**: Geographic information that can be used to map and visualize the data.

This dataset provides a comprehensive look into school safety, allowing for detailed analysis and reporting.
