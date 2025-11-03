# Data-Driven Strategies to Reduce Employee Turnover
Final Project ini disusun sebagai salah satu syarat untuk menyelesaikan Data Science Career Bootcamp di <a href="https://www.rakamin.com/">Rakamin</a>.

<p align="center">
  <img src="Images/banner-by-namogoo.png" width="1024" height="auto">
  <br>
</p>

## Authors and Contributors
* Michael Timotius as Project Manager
* William Wijaya as Business Analyst
* Fani Dwi as Data Scientist
* Taufiq Hidayatullah as Data Engineer

## Table of Contents
* [Preparation](#Stage-0-Preparation)
    * [Problem Statement](#01-Problem-Statement)
    * [Goal](#02-Goal)
    * [Objective](#03-Objective)
* [Exploratory Data Analysis](#Stage-1-Exploratory-Data-Analysis)
    * [Data Exploration](#11-Data-Exploration)
    * [Data Understanding](#12-Data-Understanding)
    * [Exploratory Data Analysis](#13-Exploratory-Data-Analysis)    
* [Data Preprocessing](#Stage-2-Data-Preprocessing)
    * [Data Cleansing](#21-Data-Cleansing)
    * [Feature Engineering](#22-Feature-Engineering)

## Stage 0. Preparation

### 0.1. Problem Statement
Dataset ini diasumsikan milik dari sebuah perusahaan asuransi yang memiliki agen perusahaan dan kantor cabang yang tersebar di seluruh wilayah Indonesia. Dataset ini memuat informasi terkait karyawan divisi sales yang memutuskan untuk keluar dari perusahaan atau tidak. Berdasarkan data yang tersedia tingkat karyawan yang memutuskan untuk keluar dari perusahaan cukup tinggi yakni sebesar *63,17%*. 

Tingginya persentase karyawan yang keluar menjadi hambatan bagi perusahaan untuk meningkatkan pendapatan karena terjadi pada divisi sales. Banyaknya karyawan yang keluar juga menambah beban keuangan perusahaan karena harus melakukan rekrutmen ulang serta memberikan pelatihan bagi karyawan baru.

### 0.2. Goal
Melalui proyek ini diharapkan dapat menghasilkan sebuah *Machine Learning* yang dapat memprediksi dan mengidentifikasi karyawan yang akan memutuskan keluar dari perusahaan.
Sehingga diharapkan perusahan dapat langsung melakukan intervensi dengan tujuan mencegah karyawan tersebut keluar dari perusahaan.

### 0.3. Objective
1. Dapat mengidentifikasi minimal 70% dari karyawan yang memutuskan untuk keluar dari perusahaan.
2. Memprediksi kapan karyawan akan memutuskan keluar dari perusahaan dengan tingkat akurasi minimal 55%.
3. Mengetahui faktor-faktor apa saja yang berpengaruh terhadap keputusan karyawan untuk keluar dari perusahaan.

## Stage 1. Exploratory Data Analysis

### 1.1. Data Exploration
Dataset [employee_churn_prediction_updated] merupakan dataset dengan total 10.000 karyawan divisi sales. Dataset ini terdiri dari 12.330 baris dan 16 kolom fitur, setiap baris berisi data yang berkaitan dengan kolom fitur target yakni karyawan keluar perusahaan atau tidak *(Churn)*.

### 1.2. Data Understanding
#### 1.2.0. Features Definition
##### 1.2.0.1. Numerical Features
| Feature Name              | Feature Description                                                    |
|:--------------------------|:-----------------------------------------------------------------------|
| `employee_id`             | Unique ID for each employee                                            |
| `age`                     | Employee's age                                                         |
| `experience_years`        | Total years of work experience                                         |
| `monthly_target`          | Monthly performance target set by the company                          |
| `target_achievement`      | Percentage of monthly target achieved by the employee                  |
| `working_hours_per_week`  | Average working hours per week                                         |
| `overtime_hours_per_week` | Number of overtime hours per week                                      |
| `salary`                  | Employee's base salary                                                 |
| `commission_rate`	        | Percentage of commission earned                                        |
| `job_satisfaction`        | Level of employee's job satisfaction                                   |
| `manager_support_score`   | Score reflecting the level of support provided by the manager          |
| `company_tenure_years`    | Number of years the employee has worked in the company                 |
| `distance_to_office_km`   | Distance from employee's home to the office (in km)                    |

##### 1.2.0.2. Categorical Features
| Feature Name              | Feature Description                                                    |
|:--------------------------|:-----------------------------------------------------------------------|
| `gender`                  | Employee's gender                                                      |
| `education`               | Employee's highest level of education                                  |
| `work_location`           | Area of employee's workplace location                                  |
| `marital_status`          | Employee's marital status                                              |

##### 1.2.0.3. Target Feature
| Feature Name              | Feature Description                                                    |
|:--------------------------|:-----------------------------------------------------------------------|
| `churn`                   | Indicates whether the employee left the company or not (target label)  |
| `churn_period`            | Time period when the employee left, if churn occurred                  |

#### 1.2.1. Data Dimension
Dataset ini memiliki dimensi data, yaitu
- Jumlah baris: 10.000
- Jumlah kolom: 18
- 
#### 1.2.2. Data Types and Structure
Untuk mendapatkan ringkasan singkat tentang dataset, kami menggunakan fungsi `info()`. Hasil observasi yang didapatkan adalah sebagai berikut.
- Seluruh kolom atau fitur sudah memiliki tipe data yang sesuai.
- Tidak ada kolom yang memiliki nilai kosong atau _missing values_.
- Tipe data berupa float (3), integer (10), dan string (5).
#### 1.2.3. Detect Missing Values
Untuk memastikan adanya _missing values_ dalam dataset, kita menggunakan metode `isna()`.
- Tidak ada kolom yang _null_ (bernilai None ataupun NaN).
#### 1.2.4. Detect Duplicates
Untuk menemukan adanya _duplicates_, kita menggunakan metode `duplicated()`. 
- Tidak ada baris data yang duplikat.
#### 1.2.5. Data Anomaly
Ditemukan 74 baris data yang memiliki nilai anomali yang memuat data karyawan dengan pengalaman kerja sangat besar atau sudah mulai bekerja sebelum berusia 18 tahun.
Meskipun memungkinkan, data ini sangatlah tidak wajar dalam perusahaan profesional sehingga diputuskan untuk dihapus. 
#### 1.2.6 Feature Creation
Membuat fitur-fitur baru dari fitur yang telah tersedia dari dataset.
- `Overtime Ratio` merupakan fitur yang menghitung rasio waktu lembur karyawan dari total jam kerja dalam seminggu.
- `Burnout Score` merupakan fitur yang melihat tingkat kelelahan karyawan dengan memperhitungkan waktu lembur, jam kerja dan kepuasaan karyawan.
- `Achievment Status` merupakan fitur yang melihat apakah karyawan mencapai target atau tidak.
- `Achievment Stress` merupakan fitur yang menghitung tingkat stress karyawan dengan memperhitungkan waktu lembur dan target.
- `Loyalty Index` merupakan fitur yang berisikan lama kerja di perusahaan dari total pengalaman yang dipunya.
- `Promotion Potential` merupakan fitur yang menghitung apakah seharusnya karyawan mendaparkan promosi berdasarkan kepuasan kerja, pemenuhan target dan lama kerja di perusahaan.
- `Distance Stress Adjusted` merupakan fitur yang menghitung indeks stress berdasarkan jarak kerja berdasarkan dukungan manajer dan status pernikahan.
- `Stress Index` merupakan fitur yang mengkombinasikan fitur burnout, achievement stress dan kelelahan akibat jarak kerja. 

### 1.3. Descriptive Statistics
Untuk mendapatkan perincian statistik dasar dari dataset, kita menggunakan metode `describe()`.
#### 1.3.1. Numerical Features
Dalam fitur numerikal, distribusi data sangat normal dan tidak ada outlier yang ditemukan. 
#### 1.3.2. Target Feature
Fitur `churn` digunakan sebagai _target feature_ atau label kelas.
- Dari total 10.000 karyawan, sebanyak 63,17% merupakan **kelas positif** yakni yang memutuskan keluar dari perusahaan, sedangkan 36,83% sisanya merupakan **kelas negatif** yakni karyawan yang bertahan di perusahaan.
- Dataset _imbalance_ atau tidak seimbang, karena proporsi data minoritas (dalam hal ini kelas negatif) relatif rendah, dengan _degree of imbalance_: [moderate](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data/).
- Pada saat data pre-processing, kita perlu melakukan _handling imbalance data_, seperti
  - Oversampling: menduplikasi data minoritas,
  - Undersampling: menghapus data mayoritas, atau
  - Class weight.

### 1.4. Univariate Analysis
_Univariate analysis_ dilakukan untuk melihat distribusi data dari setiap fitur secara terpisah. Bisa melihat distribusi data apakah berdistribusi normal, _left skew_, atau _right skew_ dengan menggunakan kdeplot. Kemudian melihat ada berapa banyak _outlier_ yang ada pada setiap fitur dengan menggunakan boxplot.
#### 1.4.1. Data Distribution
Dari distribusi data dapat disimpulkan bahwa terdapat pola beberapa puncak distribusi (multimodal pattern/many peaks). Ini menandakan bahwa karakteristik data sangat beragam dan terdiri dari beberapa kelompok populasi yang tidak seragam. 

Pada saat data pre-processing, kita perlu:
- Melakukan _Feature Transformation_ dengan _Standarization_, karena terdapat banyak fitur yang memiliki sebaran nilai atau skala yang berbeda.
- Melakukan _Feature Encoding_ untuk fitur `Gender`, `Education`, `work_location` dan `marital_status` menggunakan _Label Encoding_.
- Melakukan _Handling Imbalanced Data_ untuk fitur `churn`, karena fitur ini merupakan target yang mempunyai ketimpangan data yang signifikan.

### 1.5. Multivariate Analysis
Untuk melakukan _multivariate analysis_ bisa menggunakan pairplot. Pairplot digunakan untuk menganalisis antara dua variabel pada data (bivariate analysis).
Berdasarkan visualisasi yang telah dilakukan terlihat bahwa karyawan yang keluar dari perusahaan dipengaruhi oleh faktor eksternal yang terdiri dari stress dan jarak kerja. Sementara untuk faktor internal dipengaruhi oleh kelelahan kerja dan loyalitas, beberapa juga dipengaruhi oleh buruknya manajerial dan kinerja yang rendah dan tidak mencapai target.

Urutan faktor yang mempengaruhi karyawan keluar dari perusahaan adalah:
- Situasi kerja: Burnout Score, Loyalty Index, Stress Index, Job Satisfaction dan Manager Support Score merupakan fitur yang paling berpengaruh.
- Kinerja: Target Achievement (gagal mencapai target) dan low Promotion Potential juga menjadi faktor yang signifikan.
- Tekanan dari luar: Distance-related stress dan high Working Hours juga berkontribusi terhadap karyawan yang keluar.

#### 1.5.1. Data Correlation
Dari korelasi data dapat disimpulkan bahwa:
- Korelasi positif: Fitur dengan korelasi positif menandakan semakin meningkatkan nilai tersebut maka peluang karyawan untuk keluar juga bertambah. Fitur ini terdiri dari:
 
1. Stress Index (0.25)
2. Distance Stress Adjusted (0.23) 
3. Achievement Stress (0.20)
4. Distance to Office (0.18)
5. Working Hours per Week (0.17)

Beban kerja dan faktor yang menyebabkan stress memiliki korelasi paling tinggi. Hal ini menandakan pentingnya keseimbangan kerja dan mental dari karyawan.

- Korelasi negatif: Fitur dengan korelasi negatif menandakan semakin meningkatkan nilai tersebut maka peluang karyawan untuk bertahan. Fitur ini terdiri dari:

1. Target Achievement (-0.30) 
2. Promotion Potential (-0.28)
3. Job Satisfaction (-0.20)
4. Marital Status - Single (-0.16)
5. Manager Support Score (-0.15)
6. Company Tenure (-0.10)

Faktor yang berkaitan dengan perkembangan karir, tercapainya target dan kepuasan kerja menjadi faktor yang membuat karyawan bertahan. 

## Stage 2. Data Preprocessing
Tahap Pengerjaan

### 2.1. Data Cleansing
### 2.1.1. Handle Missing Values
- Pada saat dilakukan handle missing values didapatkan jumlah nilai pada dataset adalah 0, sehingga dapat disimpulkan dataset ini bersih karena tidak memiliki nilai kosong
### 2.1.2. Handle Duplicated Data
- Pada saat dilakukan Handle ducpliated data didapatkan jumlah nilai pada dataset adalah 0, sehingga dapat disimpulkan dataset ini bersih karena tidak memiliki nilai kosong
### 2.1.3. Handle Outliers
- Pada saat dilakukan pengecekan distribusi data tidak ditemukan fitur dan target yang memiliki outlier, sehingga tidak diperlakukan handling outliers.
### 2.1.4. Feature Transformation
- Kita menggunakan Standarization untuk mengubah fitur numerikal agar memiliki skala nilai yang sama besar.
### 2.1.5. Feature Encoding
Kita akan melakukan _feature encoding_ terhadap fitur `VisitorType`.
- Terdapat 18 Fitur sebelum encoding dan jumlah fitur sesudah encoding sebanyak 27 Fitur.

### 2.1.6. Handle Class Imbalance
Kami menggunakan metode Random Over-Sampling untuk _handle_ fitur target yang tidak seimbang dengan menambahkan jumlah sample pada minority class sehingga setara dengan majority class
- Dari proses pengerjaan yang kami lakukan, kami menemukan bahwa jumlah karyawan yang keluar sebanyak 467 dibanding yang bertahan 273.
- Setelah melakukan proses handling terhadap imbalance class, data karyawn yang keluar dan bertahan masing-masing berjumlah 467.

## License

The source code for the site is licensed under the MIT license, which you can find [here](https://github.com/sabirinID/Final-Project-Quattro/blob/main/LICENCE).

## References

