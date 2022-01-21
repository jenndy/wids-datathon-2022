import streamlit as streamlit
import pandas as pandasimport matplotlib.pyplot as pyplot
import seaborn as sns
import import

@st.cache
def get_data():
    df = pd.read_csv('train.csv')
    # df['site_eui_per_sqft'] = df['site_eui'] / df['floor_area']
    return df

@st.cache
def get_missing_table(train_df):
    missing_columns = [col for col in train_df.columns if train_df[col].isnull().any()]
    missingvalues_count = train_df.isna().sum()
    missing_df = pd.DataFrame(missingvalues_count.rename('Null Values Count').loc[missingvalues_count.ne(0)])
    return missing_df

train_df = get_data()
numerical_features = [
    'Year_Factor',
    'floor_area',
    'year_built',
    'energy_star_rating',
    'ELEVATION',
    'cooling_degree_days',
    'heating_degree_days',
    'precipitation_inches',
    'snowfall_inches',
    'snowdepth_inches',
    'avg_temp',
    'site_eui'
]
categorical_features = ['State_Factor', 'building_class', 'facility_type']

st.sidebar.title("WiDS 2022 Datathon")
st.sidebar.header("Analyze Building Data")
option = st.sidebar.selectbox(
    'Select an Option',
    (
        'Data Overview',
        'Data Summary',
        'Missing or Extreme Values',
        'Categorical Variable Distributions',
        'Numeric Variable Distributions',
        'Target Variable Viz'
    )
)
st.title(option)

if option == "Data Summary":
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(train_df), None)
    col2.metric("Columns", len(train_df.columns), None)
    # col3.metric("", None, None)

    st.subheader('Sample of Dataset')
    st.dataframe(train_df.head(5))

    st.subheader('Dataset Summary')
    st.dataframe(train_df.describe())

    st.subheader('Dataset Information')
    buffer = io.StringIO()
    train_df.info(buf = buffer)
    st.text(buffer.getvalue())

elif option == "Data Overview":
    st.markdown(
        """
        ### Data Dictionary
        #### Original Data
        - `id`: building id
        - `Year_Factor`: anonymized year in which the weather and energy usage factors were observed
            - looks like there are 6 different years
        - `State_Factor`: anonymized state in which the building is located
            - looks like there are 7 unique states
        - `building_class`: building classification
            - Either `Commercial` or `Residential`
        - `facility_type`: building usage type (60 different types - is it grocery store? office building etc)
        - `floor_area`: floor area (in square feet) of the building
        - `year_built`: year in which the building was constructed
            - there is a `0` value? ranges from 1732 (early 1700s) to 2015?
        - `energy_star_rating`: the energy star rating of the building
            - value between 0-100. seems like there are some missing values?
        - `ELEVATION`: elevation of the building location
        - `january_min_temp`: minimum temperature in January (in Fahrenheit) at the location of the building
        - `january_avg_temp`: average temperature in January (in Fahrenheit) at the location of the building
        - `january_max_temp`: maximum temperature in January (in Fahrenheit) at the location of the building
        - `cooling_degree_days`: cooling degree day for a given day is the number of degrees where the daily average temperature exceeds 65 degrees Fahrenheit. Each month is summed to produce an annual total at the location of the building.
        - `heating_degree_days`: heating degree day for a given day is the number of degrees where the daily average temperature falls under 65 degrees Fahrenheit. Each month is summed to produce an annual total at the location of the building.
        - `precipitation_inches`: annual precipitation in inches at the location of the building
        - `snowfall_inches`: annual snowfall in inches at the location of the building
        - `snowdepth_inches`: annual snow depth in inches at the location of the building
        - `avg_temp`: average temperature over a year at the location of the building
        - `days_below_30F`: total number of days below 30 degrees Fahrenheit at the location of the building
        - `days_below_20F`: total number of days below 20 degrees Fahrenheit at the location of the building
        - `days_below_10F`: total number of days below 10 degrees Fahrenheit at the location of the building
        - `days_below_0F`: total number of days below 0 degrees Fahrenheit at the location of the building
        - `days_above_80F`: total number of days above 80 degrees Fahrenheit at the location of the building
        - `days_above_90F`: total number of days above 90 degrees Fahrenheit at the location of the building
        - `days_above_100F`: total number of days above 100 degrees Fahrenheit at the location of the building
        - `days_above_110F`: total number of days above 110 degrees Fahrenheit at the location of the building
        - `direction_max_wind_speed`: wind direction for maximum wind speed at the location of the building. Given in 360-degree compass point directions (e.g. 360 = north, 180 = south, etc.).
        - `direction_peak_wind_speed`: wind direction for peak wind gust speed at the location of the building. Given in 360-degree compass point directions (e.g. 360 = north, 180 = south, etc.).
        - `max_wind_speed`: maximum wind speed at the location of the building
        - `days_with_fog`: number of days with fog at the location of the building

        #### Target
        - `site_eui`: Site Energy Usage Intensity is the amount of heat and electricity consumed by a building as reflected in utility bills
        """
    )

elif option == "Missing or Extreme Values":
    st.subheader('Heatmap of Missing Values in Training Data')
    fig = plt.figure(figsize = (10, 4))
    sns.heatmap(train_df.isna(), cmap = ['navy', 'yellow'])
    st.pyplot(fig, use_container_width = True)

    st.dataframe(get_missing_table(train_df))

    st.subheader("Checking the `year_built`")
    st.write("After we drop missing/null values, what is the distribution of `year_built`?")
    st.write('Looks like we have 0.0 as a year?!')
    fig = plt.figure(figsize = (10, 4))
    sns.kdeplot(train_df['year_built'].values)
    sns.rugplot(train_df['year_built'].values)
    st.pyplot(fig, use_container_width = True)

elif option == "Numeric Variable Distributions":
    selected_column = st.selectbox(
        "Select a column to visualize",
        numerical_features
    )
    st.subheader("Analyzing `" + selected_column + "`")
    fig = plt.figure(figsize = (10, 4))
    sns.kdeplot(train_df[selected_column].values)
    sns.rugplot(train_df[selected_column].values)
    st.pyplot(fig, use_container_width = True)

elif option == "Categorical Variable Distributions":
    selected_column = st.selectbox(
        "Select a column to visualize",
        categorical_features
    )
    st.subheader("Analyzing `" + selected_column + "`")
    fig = plt.figure(figsize = (10, 4))
    ax = sns.countplot(train_df[selected_column].values)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    sns.rugplot(train_df[selected_column].values)
    st.pyplot(fig, use_container_width = True)

elif option == "Target Variable Viz":
    selected_xaxis = st.selectbox(
        "select a feature for the x-axis",
        categorical_features + numerical_features
    )

    selected_color = st.selectbox(
        "select a feature to color by",
        categorical_features
    )

    fig = plt.figure(figsize = (10, 4))
    st.subheader("Understanding `site_eui` and `" + selected_xaxis + "`")
    sns.scatterplot(data = train_df, y = "site_eui", x = selected_xaxis, s = 5, hue = selected_color)
    st.pyplot(fig, use_container_width = True)