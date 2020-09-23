# Personal Data Science Projects

## [KA-Feedback](KA-Feedback/)

KA-Feedback is a system that allows inhabitants of Karlsruhe to notify the municipality of certain situations in the public space. The data is analysed and presented. (In German.)

__Result__: Highlights of interesting features in the data.

__Skills__: `data preprocessing`, `data visualisation`

## [Microclimate Forecaster](MicroclimateForecaster/)

A temperature sensor connected to a Raspberry Pi is used to display the living room temperature on 4.2" e-Paper display. This project takes the next step by forecasting the near-future temperature based on historical values. A combination of mathematical models and (recurrent) neural networks is explored to perform time-series forecasting.

__Result__: An error of 5.6% on the temperature forecasting at a depth of ten time steps into the future.

__Skills__: `time-series analysis`, `recurrent neural network`, `tensorflow`

## [Scientific Paper Classifier](ScientificPaperClassifier/)

Scientists publish papers. The arXiv.org organises preprints in various categories, but there is a thin line between two of those categories: `hep-ph` and `hep-th`, which stand for high energy physics (hep) - phenomenology (ph) and theory (th). Many researchers submit some articles to `hep-ph` and some others to `hep-th`, but sometimes the choice is not so clear. This project employs machine learning to build a paper classifier to provide a definitive way out in such a dilemma.

__Result__: A balanced classes binary classification accuracy of 86%.

__Skills__: `web scraping`, `natural language processing`, `scikit-learn`

## [Laptop Battery Monitor](LaptopBatteryMonitor/)

MacOS provides four stages of battery condition: "Normal", "Replace Soon", "Replace Now" and "Service Battery". But just how exactly is the battery performing? How much charge is it holding? How fast is it going through the re-charging cycles? This project answers such questions by gathering and visualising data on the laptop battery status, in order to provide a deeper insight than MacOS's built-in battery condition message.

__Result__: A report on historic data for three battery KPIs that can be displayed in a web-browser.

__Skills__: `data processing`, `data visualisation`