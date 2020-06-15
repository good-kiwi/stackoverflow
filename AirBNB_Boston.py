import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
from scipy import stats
import math


def clean_data(df):
    """
    INPUT
    df - pandas dataframe

    OUTPUT
    X - A matrix holding all of the variables you want to consider when predicting the response
    y - the corresponding response vector

    This function cleans df using the following steps to produce X and y:
    1. Drop rows with 0 price and outlier prices (prices above 2950)
    2. Create y as the price column, transformed by log
    3. Create X from selected columns
    4. Deal with missing values
    5. Create dummy variables for selected categorical variables, drop the original columns
    """

    # Drop rows with 0 price
    df = df[df.price > 0]
    df = df[df.price < 2950]

    # Create y
    y = df['price'].apply(math.log)

    # Select columns for X
    potential_vars = ['host_listings_count',
                     'calculated_host_listings_count_private_rooms',
                     'neighbourhood_cleansed',
                     'room_type',
                     'property_type',
                     'beds',
                     'availability_365',
                     'number_of_reviews',
                     'neighborhood_overview',
                     'space',
                     'notes',
                     'transit',
                     'access',
                     'interaction',
                     'house_rules',
                     'host_about',
                     'host_is_superhost',
                     'host_has_profile_pic',
                     'host_identity_verified',
                     'instant_bookable',
                     'require_guest_profile_picture',
                     'require_guest_phone_verification',]

    bool_vars = ['host_is_superhost',
                 'host_has_profile_pic',
                 'host_identity_verified',
                 'instant_bookable',
                 'require_guest_profile_picture',
                 'require_guest_phone_verification']

    free_text_vars = ['neighborhood_overview',
                      'space',
                      'notes',
                      'transit',
                      'access',
                      'interaction',
                      'house_rules',
                      'host_about']

    df = df[potential_vars]
    # Deal with missing values
    df['number_of_reviews'].fillna(0, inplace=True)
    df[bool_vars].fillna('f', inplace=True)
    df[free_text_vars].fillna('', inplace=True)

    def translate_bool(col):
        for index, value in col.iteritems():
            col[index] = 1 if value == 't' else 0
        return col

    def create_bool(col):
        for index, value in col.iteritems():
            col[index] = 0 if value == '' else 1
        return col

    fill_mean = lambda col: col.fillna(col.mean())
    num_vars = df.select_dtypes(include=['int', 'float']).columns
    df[num_vars] = df[num_vars].apply(fill_mean, axis=0)
    df[bool_vars] = df[bool_vars].apply(translate_bool, axis=0)
    df[bool_vars].dtype = int
    df[free_text_vars] = df[free_text_vars].apply(create_bool, axis=0)
    df[free_text_vars].dtype = int
    # Dummy the categorical variables
    cat_vars = ['neighbourhood_cleansed', 'room_type', 'property_type']
    for var in cat_vars:
        # for each cat add dummy var, drop original column
        df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=True)], axis=1)

    X = df

    return X, y


def find_optimal_lm_mod(X, y, cutoffs, test_size = .30, random_state=42, plot=True):
    '''
    INPUT
    X - pandas dataframe, X matrix
    y - pandas dataframe, response variable
    cutoffs - list of ints, cutoff for number of non-zero values in dummy categorical vars
    test_size - float between 0 and 1, default 0.3, determines the proportion of data as test data
    random_state - int, default 42, controls random state for train_test_split
    plot - boolean, default 0.3, True to plot result

    OUTPUT
    r2_scores_test - list of floats of r2 scores on the test data
    r2_scores_train - list of floats of r2 scores on the train data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()
    for cutoff in cutoffs:

        #reduce X matrix
        reduce_X = X.iloc[:, np.where((X.sum() > cutoff) == True)[0]]
        num_feats.append(reduce_X.shape[1])

        #split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

        #fit the model and obtain pred response
        lm_model = LinearRegression(normalize=True)
        lm_model.fit(X_train, y_train)
        y_test_preds = lm_model.predict(X_test)
        y_train_preds = lm_model.predict(X_train)

        #append the r2 value from the test set
        r2_scores_test.append(r2_score(y_test, y_test_preds))
        r2_scores_train.append(r2_score(y_train, y_train_preds))
        results[str(cutoff)] = r2_score(y_test, y_test_preds)

    if plot:
        plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5)
        plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)
        plt.xlabel('Number of Features')
        plt.ylabel('Rsquared')
        plt.title('Rsquared by Number of Features')
        plt.legend(loc=1)
        plt.show()

    best_cutoff = max(results, key=results.get)

    #reduce X matrix
    reduce_X = X.iloc[:, np.where((X.sum() > int(best_cutoff)) == True)[0]]
    num_feats.append(reduce_X.shape[1])

    #split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    #fit the model
    lm_model = LinearRegression(normalize=True)
    lm_model.fit(X_train, y_train)

    return r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test


def main():
    plot = False  # set to true if you would like to see plots
    print_log = True  # set to true if you would like to see stats outputted to console
    print_result = True

    # Data Exploration
    desired_width=320
    pd.set_option('display.width', desired_width)
    pd.set_option('display.max_columns', 50)

    # Get a sense of the numerical data in the available datasets.
    df_listings = pd.read_csv('data/listings_boston.csv', dtype={"price": str,
                                                                 "weekly_price": str,
                                                                 "monthly_price": str,
                                                                 "security_deposit": str,
                                                                 "cleaning_fee": str,
                                                                 "extra_people": str,
                                                                 "host_response_rate": str})
    # clean up price data to make it numeric
    df_listings.loc[:, "price"] = df_listings["price"].str.replace(',', '').str.replace('$', '').astype('float')
    df_listings.loc[:, "weekly_price"] = df_listings["weekly_price"].str.replace(',', '').str.replace('$', '').astype('float')
    df_listings.loc[:, "monthly_price"] = df_listings["monthly_price"].str.replace(',', '').str.replace('$', '').astype('float')
    df_listings.loc[:, "security_deposit"] = df_listings["security_deposit"].str.replace(',', '').str.replace('$', '').astype('float')
    df_listings.loc[:, "cleaning_fee"] = df_listings["cleaning_fee"].str.replace(',', '').str.replace('$', '').astype('float')
    df_listings.loc[:, "extra_people"] = df_listings["extra_people"].str.replace(',', '').str.replace('$', '').astype('float')
    df_listings["host_response_rate"].fillna("0", inplace=True)
    df_listings.loc[:, "host_response_rate"] = df_listings["host_response_rate"].str.replace('%', '').astype('int')
    if print_log:
        print(df_listings.describe())
    df_neighborhoods = pd.read_csv('data/neighbourhoods_boston.csv')
    if print_log:
        print(df_neighborhoods.describe())
    df_reviews = pd.read_csv('data/reviews_boston.csv')
    if print_log:
        print(df_reviews.describe())
    df_calendar = pd.read_csv('data/calendar_boston.csv', dtype={"price": str, "adjusted_price": str})
    # clean up price data to make it numeric
    df_calendar.loc[:, "price"] = df_calendar["price"].str.replace(',', '').str.replace('$', '').astype('float')
    df_calendar.loc[:, "adjusted_price"] = df_calendar["adjusted_price"].str.replace(',', '').str.replace('$', '').astype('float')
    if print_log:
        print(df_calendar.describe())

    # df_neighborhoods is basically empty and can be ignored
    # df_reviews is full of unstructured review data and would have to be mined to produce modelable data
    # df_listings has descriptive information about the location
    # df_calendar has price information and how it varies over time. Price and adjusted price have to be formatted.
    # Going to primarily focus on df_listings

    # How many N/A values are there for each column?
    if print_log:
        for col in df_listings.columns:
            print(col, ':', df_listings[col].dropna().shape[0] / df_listings[col].shape[0])

    # Possible binary variable conversions: neighborhood_overview, space, notes, transit, access, interaction,
    # house_rules

    # Are there any correlations we should worry about?
    num_vars = ["price",
                "weekly_price",
                "monthly_price",
                "security_deposit",
                "cleaning_fee",
                "extra_people",
                'host_listings_count',
                'host_total_listings_count',
                'calculated_host_listings_count',
                'calculated_host_listings_count_entire_homes',
                'calculated_host_listings_count_private_rooms',
                'calculated_host_listings_count_shared_rooms',
                'host_response_rate',
                'accommodates',
                'bathrooms',
                'bedrooms',
                'beds',
                'square_feet',
                'guests_included',
                'minimum_nights',
                'minimum_minimum_nights',
                'maximum_minimum_nights',
                'minimum_nights_avg_ntm',
                'maximum_nights',
                'minimum_maximum_nights',
                'maximum_maximum_nights',
                'maximum_nights_avg_ntm',
                'availability_30',
                'availability_60',
                'availability_90',
                'availability_365',
                'number_of_reviews',
                'number_of_reviews_ltm',
                'reviews_per_month',
                'review_scores_rating',
                'review_scores_accuracy',
                'review_scores_cleanliness',
                'review_scores_checkin',
                'review_scores_communication',
                'review_scores_location',
                'review_scores_value'
                ]
    if plot:
        sns.heatmap(df_listings[num_vars].corr(), annot=False, fmt=".2f", cmap="YlGnBu", linewidths=.5, square=True)
        plt.show()

    # Correlation matrix supports some clearly distinct categories of data
    # Pricing: price, weekly_price, monthly_price, security_deposit, cleaning_fee, extra_people
    # Host: host_listings_count, host_total_listings_count, calculated_host_listings_count,
    #  calculated_host_listings_count_entire_homes, calculated_host_listings_count_private_rooms,
    #  calculated_host_listings_count_shared_rooms
    # Property: accommodates, bathrooms, bedrooms, beds, square_feet, guests_included, minimum_nights,
    #  minimum_minimum_nights, maximum_minimum_nights, minimum_nights_avg_ntm, maximum_nights, minimum_maximum_nights,
    #  maximum_maximum_nights, maximum_nights_avg_ntm
    # Availability: availability_30, availability_60, availability_90, availability_365
    # Reviews: number_of_reviews, number_of_reviews_ltm, reviews_per_month, review_scores_rating,
    #  review_scores_cleanliness, review_scores_checkin, review_scores_communication, review_scores_location,
    #  review_scores_value

    # Get a sense of the categorical data in the available data.
    cat_vars = ["space",
                "description",
                "experiences_offered",
                "neighborhood_overview",
                "notes",
                "transit",
                "access",
                "interaction",
                "house_rules",
                "host_name",
                "host_since",
                "host_location",
                "host_about",
                "host_response_time",
                "host_acceptance_rate",
                "host_is_superhost",
                "host_neighbourhood",
                "host_verifications",
                "host_has_profile_pic",
                "host_identity_verified",
                "street",
                "neighbourhood",
                "neighbourhood_cleansed",
                "market",
                "smart_location",
                "is_location_exact",
                "property_type",
                "room_type",
                "bed_type",
                "amenities",
                "extra_people",
                "calendar_updated",
                "has_availability",
                "calendar_last_scraped",
                "requires_license",
                "instant_bookable",
                "is_business_travel_ready",
                "cancellation_policy",
                "require_guest_profile_picture",
                "require_guest_phone_verification"]

    if print_log:
        for col in df_listings[cat_vars].columns:
            print(df_listings[[col, 'price']].groupby([col]).mean())
            print(df_listings[col].value_counts())

    # free text columns: space, description, neighborhood_overview, notes, transit, access, interaction, house_rules,
    #  host_name, host_about,
    # empty: experiences_offered, market, calendar_last_scraped, requires_license, is_business_travel_ready
    # boolean: host_is_superhost, host_has_profile_pic, host_identity_verified, is_location_exact, has_availability,
    #  instant_bookable, require_guest_profile_picture, require_guest_phone_verification, host_about
    # categorical: property_type, room_type, bed_type, amenities, calendar_updated, cancellation_policy,

    if print_log:
        print(pd.crosstab(df_listings['neighbourhood'], df_listings['room_type']))
    # Surprised to see that the top neighborhoods are not very desirable areas to vacation in.
    # Also the majority of the listings are for an entire unit.

    # Explore target variable, price
    target = df_listings['price'].copy()
    if plot:
        plt.hist(target, bins=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
        plt.show()
    # how many zero values are there?
    if print_log:
        print("Number of 0 price listings: ", (target == 0).sum())

    # Need to drop 0 price listings
    target = target[target != 0]

    # Drop outlier prices
    # target = target[target <= 4000]

    # seems like a log-normal distribution
    shape, loc, scale = stats.lognorm.fit(target)
    if print_log:
        print(shape, loc, scale)
        print(stats.kstest(target, "lognorm", args=[shape, loc, scale]))
    if plot:
        sns.distplot(target, fit=stats.lognorm, kde=False, rug=True)
        plt.show()
        linspace = np.linspace(0, 1000, 100)
        pdf_lognorm = stats.lognorm.pdf(linspace, shape, loc, scale)
        plt.hist(target, density=True, bins=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
        plt.plot(linspace, pdf_lognorm, label="lognorm")
        plt.show()
    X, y = clean_data(df_listings)
    print(X.head())
    print(y.head())
    r2_scores_test, r2_scores_train, lm_model, X_train, X_test, y_train, y_test = find_optimal_lm_mod(X, y, [100, 1000, 2000, 3000], plot=plot)
    if print_log:
        print(list(zip(X_train.columns, np.exp(lm_model.coef_))))
        print(r2_scores_train, r2_scores_test)

    selected_vars=X_train.columns

    lm_model = LinearRegression(normalize=False)
    X = X[selected_vars]
    lm_model.fit(X, y)
    y_preds = lm_model.predict(X)

    if print_result:
        print(list(zip(X.columns, np.exp(lm_model.coef_))))
        print(r2_score(y, y_preds))
        sns.residplot(y, y_preds)
        plt.show()
    return lm_model


if __name__ == '__main__':
    best_model = main()
