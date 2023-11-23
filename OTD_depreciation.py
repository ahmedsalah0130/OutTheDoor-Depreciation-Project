import geoplot
import contextily as ctx
import geopandas as gpd
from scipy.optimize import curve_fit
from scipy.signal import convolve2d
from scipy.interpolate import griddata
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cars = pd.read_csv('vehicles.csv')
# drop cars with prices above $100,000 and below $100
cars = cars[cars.price < 1e5]
cars = cars[cars.price > 100]
cars = cars[['id', 'price', 'year', 'manufacturer', 'model', 'odometer',
             'type', 'lat', 'long']]  # we will only look at these columns in the analysis
cars['age'] = 2020 - cars['year']  # add an age column
cars.columns

# Missing data


def get_missing_info(df):
    num_entries = df.shape[0]*df.shape[1]
    null_entries = df.isnull().sum().sum()
    percent_empty = null_entries/num_entries*100
    num_missing = df.isna().sum()
    percent_missing = num_missing/len(df)*100
    col_modes = df.mode().loc[0]
    percent_mode = [df[x].isin(
        [df[x].mode()[0]]).sum()/len(df)*100 for x in df]
    missing_value_df = pd.DataFrame({'num_missing': num_missing,
                                     'percent_missing': percent_missing,
                                     'mode': col_modes,
                                     'percent_mode': percent_mode})
    print('total empty percent:', percent_empty, '%')
    print('columns that are more than 97% mode:',
          missing_value_df.loc[missing_value_df['percent_mode'] > 97].index.values)
    return (missing_value_df)


get_missing_info(cars)


def modef(x):  # get mode of groupby row
    m = pd.Series.mode(x)
    if len(m) == 1:
        return m
    if len(m) == 0:
        return 'unknown'
    else:
        return m[0]


def isnan(x):  # check if entry is nan
    try:
        out = math.isnan(float(x))
    except:
        out = False
    return (out)


def fill_type(x):  # fill type column with mode of model columns
    if isnan(x['type']):
        try:
            out = model_types[x['model']]
        except:
            out = 'unknown'
    else:
        out = x['type']
    return (out)


model_types = cars.groupby(['model'])['type'].agg(modef)
cars['type'] = cars.apply(fill_type, axis=1)

# plot price histogram
sns.displot(cars, x='price', binwidth=1000,
            height=5, aspect=2)  # , bw_adjust=0.4)
plt.xticks(range(0, int(1e5), int(1e4)))
plt.xlabel('Price ($)')
plt.title('Price Distribution of Vehicles (Under $100,000)')
plt.show()

# plot pricing probability density for different types of vehicle
cars_plt = cars[cars.type.isin(['sedan', 'SUV', 'truck'])]
sns.displot(cars_plt, x='price', hue='type', kind='kde',
            bw_adjust=0.6, cut=0, common_norm=False, height=5, aspect=2)
plt.xticks(range(0, int(1e5), int(1e4)))
plt.xlabel('Price ($)')
plt.xlim(0, int(1e5))
plt.ylabel('Normalized Probability Density')
plt.title('Price Density Distribution by Type (Under $100,000)')
plt.show()

# get contour plot of price vs odometer and year
# drop rows if odometer or year are missing
carsd = cars.dropna(axis=0, subset=['odometer', 'year'])

# interpolate pricing data using several different methods
# available sample data for the contour plot
xs = carsd['odometer']
ys = carsd['year']
zs = carsd['price']
points = np.array([xs, ys]).T
# we wish to interpolate the data above onto the grid below
grid_x, grid_y = np.meshgrid(
    # odometer goes from 0 to 300,000km with steps of 300,000/10000 = 30mi
    np.linspace(0, 3e5, 10000),
    np.arange(1970, 2021, 1))  # year goes from 1970 to 2021 in steps of 1 year

# try out three different methods of interpolation
grid_z0 = griddata(points, zs, (grid_x, grid_y), method='nearest')
grid_z1 = griddata(points, zs, (grid_x, grid_y), method='linear')
grid_z2 = griddata(points, zs, (grid_x, grid_y), method='cubic')

# plot the raw data
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs[0, 0].scatter(xs, ys, c=zs)
axs[0, 0].set_xlim(0, 6e5)
axs[0, 0].set_title('raw data')

# plot the three different interpolation methods
im = axs[0, 1].contour(grid_x, grid_y, grid_z0)
fig.colorbar(im, ax=axs[0, 1])
axs[0, 1].set_title('nearest')

im = axs[1, 0].contour(grid_x, grid_y, grid_z1)
fig.colorbar(im, ax=axs[1, 0])
axs[1, 0].set_title('linear')

im = axs[1, 1].contour(grid_x, grid_y, grid_z2)
fig.colorbar(im, ax=axs[1, 1])
axs[1, 1].set_title('cubic')
plt.show()

# Filter out noise in the interpolated dataset and plot the final contour.
# size of averaging window for odometer (500steps * 60mi/step = 30,000mi)
sz_o = 500
sz_y = 3  # size of averaging window for year (3 years or +-1 year)
# averaging kernel, corresponds to averaging over +-15000 mi and +-1 year
kernel = np.ones((sz_y, sz_o))/(sz_y*sz_o)
# run a moving average over the 'nearest' interpolated dataset
grid_z0f = convolve2d(grid_z0, kernel, boundary='symm', mode='same')

fig, ax = plt.subplots(1, figsize=(9, 7))
im = ax.contourf(grid_x, grid_y, grid_z0f, levels=15,
                 cmap='RdYlBu_r', zorder=0)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Price ($)')
ax.set_xlim(0, 3e5)
ax.set_xlabel('Odometer (mi)')
ax.set_ylabel('Year')
ax.set_title('Contours of Averaged Pricing Data')
ax.grid(True, color='k')
ax.annotate("", xy=(1.35e5, 2010), xytext=(0, 2020),
            arrowprops=dict(arrowstyle="->", color='k'))
plt.show()

# find x grid location where km's driven is what we want
xloc_e = np.where((1.349e5 < grid_x[0]) & (grid_x[0] < 1.36e5))
yloc_e = 40  # row 30 of the y grid is 2010
price_end = grid_z0f[yloc_e, xloc_e[0]]
yloc_s = 50  # row 40 of the y grid is 2020
xloc_s = np.where(grid_x[0] == 0)
price_start = grid_z0f[yloc_s, xloc_s[0]]
depr_rate = ((price_start-price_end)/1.35e5)[0]
print('Benchmark Depreciation rate: ${:.2f}/mi'.format(depr_rate))

# Take a look at the most popular vehicles for sale
# add a column with the make and model in one string (for plotting)
cars['make_model'] = cars['manufacturer'] + ': ' + cars['model']
com_cars = cars.make_model.value_counts()[:25]  # the 25 most popular cars

# plot the results
fig = com_cars.plot.bar(figsize=(12, 5))
plt.xlabel('Make and Model')
plt.ylabel('Number of Postings')
plt.title('25 Most Popular Vehichles')
plt.xticks(rotation=80)
plt.show()

# plot the average prices of the 25 most popular cars
com_price = cars.loc[cars.make_model.isin(com_cars.index)]
ordered_labels = com_price.groupby(
    'make_model').price.median().sort_values(ascending=False).index.values

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=com_price, x="make_model",
            y="price", order=ordered_labels, ax=ax)
plt.xticks(rotation=80)
plt.xlabel('Make and Model')
plt.ylabel('Price ($)')
plt.title('Pricing of the 25 Most Popular Vehichles')
plt.show()

# plot the average prices of the 25 most popular trucks, trucks, and SUVS
for thing in ['sedan', 'truck', 'SUV']:
    com = cars[cars['type'] == thing].make_model.value_counts()[0:25].index
    com_price = cars.loc[cars.make_model.isin(com)]
    ordered_labels = com_price.groupby(
        'make_model').price.median().sort_values(ascending=False).index.values

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=com_price, x="make_model",
                y="price", order=ordered_labels, ax=ax)
    plt.xticks(rotation=80)

    plt.xlabel('Make and Model')
    plt.ylabel('Price ($)')
    plt.title('Pricing of the 25 Most Popular {}s'.format(thing.capitalize()))
    plt.show()

    # fit an exponential to the toyota corolla data to determine how well it holds its value


def func(x, a, b):
    return a * np.exp(-b*x)  # exponential function we will use to fit


def plot_depr(data, func, model):
    # get model data and filter out cars older than 50 years
    df = data[(data['make_model'] == model) & (
        data['age'] <= 50)].sort_values(by='age')
    xdata = df['age']
    ydata = df['price']

    # fit to the data
    # fit the exponential to the data
    popt, _ = curve_fit(func, xdata, ydata, p0=[4e4, 0.1])
    init = popt[0]  # intiial value (age=0) according to the curve fit
    # time to depreciate 20% according to the curve fit
    depr20 = -np.log(0.80)/popt[1]
    # time to depreciate 90% according to the curve fit
    depr90 = -np.log(0.10)/popt[1]

    fig, ax = plt.subplots(figsize=(10, 5))
    # scatter plot of age vs price, colored by odometer
    carplt = ax.scatter(xdata, ydata, c=df['odometer'], cmap='viridis')
    plt.plot(xdata, func(xdata, *popt), 'r--')  # plot the fitted curve

    plt.text(0.5, 0.85,
             'Initial Value: {:,.0f}$\n'
             'Time to lose 20% value: {:.2f} years\n'
             'Time to lose 90% value: {:.2f} years'.format(
                 init, depr20, depr90),
             transform=ax.transAxes,
             bbox=dict(facecolor='white', edgecolor='black'))

    cbar = plt.colorbar(carplt)
    cbar.set_label('Odomater (mi)')
    plt.xlabel('Age (years)')
    plt.ylabel('Price ($)')
    plt.title(model)
    plt.show()


plot_depr(cars, func, 'toyota: corolla')

# run curve fits for the 25 most popular sedans, SUVs, and trucks and plot
for kind in ['sedan', 'SUV', 'truck']:  # loop over the type of vehicle
    com = cars[cars['type'] == kind].make_model.value_counts(
    )[0:25].index  # 25 most popular models of this type
    # initialize an empty dataframe to hold the data
    depr_df = pd.DataFrame(columns={'Model', 'val0', 'depr20', 'depr90'})
    for name in com:  # loop over the models
        df = cars[(cars['make_model'] == name) & (
            cars['age'] <= 50)].sort_values(by='age')
        xdata = df['age']
        ydata = df['price']
        popt, pcov = curve_fit(func, xdata, ydata, p0=[4e4, 0.1])

        init = popt[0]
        depr20 = -np.log(0.80)/popt[1]
        depr90 = -np.log(0.10)/popt[1]
        depr_df = depr_df.append(
            {'Model': name, 'val0': init, 'depr20': depr20, 'depr90': depr90}, ignore_index=True)

    depr_df = depr_df.sort_values(by='depr20', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=depr_df, x='Model', y='depr20', ax=ax)
    plt.title('Depreciation of the 25 Most Popular {}s'.format(kind.capitalize()))
    plt.ylabel('Time to Depreciate 20% (Years)')
    plt.xticks(rotation=80)
    plt.show()

    # Boxplot of deprecition for different manufactureers
# 15 most popular manufacturers
makes = cars['manufacturer'].value_counts()[:15].index
# this will hold the depreciation data
depr_df = pd.DataFrame(columns={'Make', 'Model', 'val0', 'depr20', 'depr90'})
for make in makes:  # loop over manufactureres
    # get the 10 most poopular models by the manufacturer
    com = cars[cars['manufacturer'] == make].model.value_counts()[0:10].index
    for name in com:  # look over the models
        df = cars[(cars['model'] == name) & (cars['age'] < 50)].sort_values(
            by='age')  # get data for the model for ages under 50
        xdata = df['age']
        ydata = df['price']
        popt, pcov = curve_fit(func, xdata, ydata, p0=[
                               4e4, 0.1])  # fit to the data

        init = popt[0]  # initial value
        depr20 = -np.log(0.80)/popt[1]  # time to depreciate 20%
        depr90 = -np.log(0.10)/popt[1]  # time to depreciate 90%
        # append this data to the dataframe
        depr_df = depr_df.append({'Make': make, 'Model': name, 'val0': init,
                                 'depr20': depr20, 'depr90': depr90}, ignore_index=True)

# order the data in terms of decreasing median depreciation time
order = depr_df.groupby('Make')['depr20'].median(
).sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=depr_df, x='Make', y='depr20', order=order)
plt.title('Depreciation of the 15 Most Popular Makes of Vehicle')
plt.ylabel('Time to Depreciate 20% (Years)')
plt.ylim(0.5, 4)
plt.xticks(rotation=90)
plt.show()

# map distribution of vehichles
gdf = gpd.GeoDataFrame(  # convert the data to a geodataframe so it can be plotted on a map
    cars, geometry=gpd.points_from_xy(cars.long, cars.lat))
# remove data outside the geographic area of interest
gdf = gdf[(22 < gdf.lat) & (gdf.lat < 65) &
          (-144 < gdf.long) & (gdf.long < -56)]
# tell geopandas what the coordinate system of our data is
# this is the latitude/longitude system the scraped data was in
gdf = gdf.set_crs(epsg=4326)
# this is the coordinate system that the imported map is in
gdf = gdf.to_crs(epsg=3857)

# plot the geographic area where data was collected and the points of the vehicles
ax = gdf.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
ctx.add_basemap(ax, zoom=4)
plt.xlim(-1.5e7, -0.7e7)
plt.ylim(2.15e6, 6.65e6)
plt.show()

# plot the distribution of sedans, trucks, and suvs in the area around Vancouver
extent = (-1.5e7, -0.7e7, 2.6e6, 6.5e6)  # x and y limits of where we will plot

# look at sales of trucks, sedans, and SUVS
gtrucks = gdf[(gdf['type'] == 'truck')]
gsedans = gdf[(gdf['type'] == 'sedan')]
gsuvs = gdf[(gdf['type'] == 'SUV')]
# get the longitude data for each type of vehicle
trucks = pd.Series(gtrucks.geometry.x)
sedans = pd.Series(gsedans.geometry.x)
suvs = pd.Series(gsuvs.geometry.x)

fig, ax = plt.subplots(2, figsize=(13, 8), gridspec_kw={
                       'height_ratios': [1, 6], 'hspace': 0})
# plot the density distribution of each type by longitde
sns.kdeplot(data=trucks, ax=ax[0], clip=(
    extent[0], extent[1]), bw_adjust=0.35, label='Trucks', color='red')
sns.kdeplot(data=sedans, ax=ax[0], clip=(
    extent[0], extent[1]), bw_adjust=0.35, label='Sedans', color='blue')
sns.kdeplot(data=suvs, ax=ax[0], clip=(
    extent[0], extent[1]), bw_adjust=0.35, label='SUVs', color='green')
# set the x limits of the plot to be the same as our map
ax[0].set_xlim(extent[0], extent[1])
ax[0].set_ylabel('Probability Density')
ax[0].set_title('Sales of Trucks and Sedans in the Lower 48 States')
ax[0].set_xticks([])
ax[0].legend()

# Plot the location of each posting on a map
geoplot.pointplot(gtrucks, ax=ax[1], s=1, color='red')
geoplot.pointplot(gsedans, ax=ax[1], s=1, color='blue')
geoplot.pointplot(gsuvs, ax=ax[1], s=1, color='green')
ax[1].axis(extent)
ctx.add_basemap(ax[1], zoom=4)

plt.show()
