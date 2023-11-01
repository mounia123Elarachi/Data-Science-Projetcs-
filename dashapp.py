import dash
from dash import dcc, html, dash_table
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import librosa
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import base64
import os
import io
import dash_bootstrap_components as dbc
import IPython.display as ipd
from dash.dash import no_update


# Load the trained model
model = load_model('LSTM1')
labelencoder = LabelEncoder()
labelencoder.fit_transform(['Airplane', 'Bikes', 'Cars', 'Helicopter',
                            'Motorcycles', 'Train', 'Truck', 'Bus', 'SportsCars', 'Emergency', 'Boats'])

# Function to predict vehicle class using MFCC features


def predict_vehicle_class_mfcc(mfcc_features):
    labelencoder = LabelEncoder()
    labelencoder.fit_transform(['Airplane', 'Bikes', 'Cars', 'Helicopter',
                               'Motorcycles', 'Train', 'Truck', 'Bus', 'SportsCars', 'Emergency', 'Boats'])
    mfccs_scaled_features = np.mean(mfcc_features.T, axis=0)
    pre = mfccs_scaled_features.reshape(1, 1, 40)
    prediction = model.predict(pre)
    classes_x = np.argmax(prediction, axis=1)
    prediction_class = labelencoder.inverse_transform(classes_x)
    return prediction_class[0]


# Function to extract MFCC features
def extract_mfcc(audio_file, segment_duration=5):
    y, sr = librosa.load(audio_file)  # Read the audio file
    # Calculate the segment length in samples
    segment_length = segment_duration * sr

    mfcc_list = []  # List to store MFCC features

    # Iterate over the audio file in segments
    for i in range(0, len(y), segment_length):
        segment = y[i:i+segment_length]  # Extract a segment of audio
        # Calculate MFCC features for the segment
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=40)
        mfcc_list.append(mfcc)  # Add MFCC features to the list

    return mfcc_list


# Set up the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "ML Project - Master DS"

# Set suppress_callback_exceptions=True in the app object
app.config["suppress_callback_exceptions"] = True

# Define the home page content
repo_link = "https://github.com/ayyoubmanssouri/vehicle-identification-based-on-sound"
link_text = "here"


home_content = html.Div(
    className='m-5',
    children=[
        html.H1("Vehicle Classification Based On Sound"),
        html.P("Welcome to the Home page of the app."),
        html.Ul(
            [
                html.Li("Problem: The problem is to identify vehicles based on sound. This is a classification problem, as there are 11 different classes of vehicles that the model must be able to distinguish between."),
                html.Li("Approach: The model uses a Long Short-Term Memory (LSTM) neural network to classify vehicle sounds. LSTMs are a type of recurrent neural network that are well-suited for tasks involving sequential data, such as audio. The model is trained on a dataset of vehicle sounds that have been labeled with their corresponding class."),
                html.Li("Features: The model is fed Mel-frequency cepstral coefficients (MFCCs) as features. MFCCs are a type of feature that is commonly used for audio classification tasks. They are extracted from the audio signal using a process called the Mel-frequency cepstral transform."),
                html.Li("Results: The model was able to achieve an accuracy of 98% on a test set of vehicle sounds. This suggests that the model is able to accurately identify vehicles based on their sound."),
            ]
        ),
        html.P(["To check our repository on github, click ",
                html.A(link_text, href=repo_link), "."])
    ]
)

# Define the page 1 content
loading_spinner = dcc.Loading(type="circle", color="#1976D2", children=[
    html.Div(id="loading-output-page1", style={"text-align": "center", "margin-top": "20px"}, children=[
        html.H3("Loading...", style={"color": "#1976D2"})
    ])
])

page1_content = html.Div(className='m-2', children=[
    html.H1("Long audio"),
    html.P("In this page, you can input a relatively large .wav file that contains sounds of multiple vehicles. The model predicts the type of each vehicle. At the end we created some visualizations based on the audio."),
    dcc.Upload(id='upload-audio-page1', children=dbc.Button('Upload Audio File',
               color="primary", className="me-1")),
    html.Div(id='output-graphs'),
    loading_spinner
])

# Define the page 2 content
loading_spinner2 = dcc.Loading(type="circle", color="#1976D2", children=[
    html.Div(id="loading-output-page2", style={"text-align": "center", "margin-top": "20px"}, children=[
        html.H3("Loading...", style={"color": "#1976D2"})
    ])
])

page2_content = html.Div(
    className='m-2',
    children=[
        html.H1("Short audio"),
        html.P("The purpose of this page is to make a prediction for a single audio vehicle. i.e, you input an audio that contains the sound of one vehicle, the model predicts what vehicle it is, and the image of the vehicle appears below."),
        dcc.Upload(
            id='upload-audio-page2',
            children=dbc.Button('Upload Audio File',
                                color="primary", className="me-1")
        ),
        html.Div(id='output-prediction'),
        loading_spinner2
    ]
)

# Define the navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/home")),
        dbc.NavItem(dbc.NavLink("Long audio", href="/page1")),
        dbc.NavItem(dbc.NavLink("Short audio", href="/page2")),
    ],
    brand="Projet ML - Master SD",
    brand_href="/home",
    color="primary",
    dark=True,
)


# Define the app layout
app.layout = html.Div(
    children=[
        navbar,
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content', className='container mt-4')]
)


# Callback to update the page content based on the URL

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/home':
        return home_content
    elif pathname == '/page1':
        return page1_content
    elif pathname == '/page2':
        return page2_content
    else:
        return home_content


# Callback for Page 1

@app.callback(
    Output('output-graphs', 'children'),
    Output('output-graphs', 'style'),
    Output('loading-output-page1', 'children'),
    Input('upload-audio-page1', 'contents'),
    State('upload-audio-page1', 'filename')
)
def update_graphs(audio_content, audio_filename):
    if audio_content is not None:
        content_type, content_string = audio_content.split(',')
        decoded = base64.b64decode(content_string)

        # Load the audio file and extract MFCC features
        mfcc_features = extract_mfcc(io.BytesIO(decoded))
        # Access the MFCC features for each segment
        class_labels = ['Airplane', 'Bikes', 'Cars', 'Helicopter', 'Motorcycles',
                        'Train', 'Truck', 'Bus', 'SportsCars', 'Emergency', 'Boats']
        class_counts = {class_label: 0 for class_label in class_labels}

        for i, mfcc in enumerate(mfcc_features):
            prediction = predict_vehicle_class_mfcc(mfcc)
            class_counts[prediction] += 1

        # Create the bar chart
        bar_chart = go.Figure(
            data=[go.Bar(x=list(class_counts.keys()),
                         y=list(class_counts.values()))],
            layout=go.Layout(
                xaxis_title='Vehicle Type',
                yaxis_title='Count',
                title=go.layout.Title(
                    text="Count Of Each Class Present In The Audio File"),
                title_font=dict(size=20),  # Adjust the font size
            )
        )

        # Create the pie chart
        total_count = sum(class_counts.values())
        pie_chart_data = [
            go.Pie(
                labels=list(class_counts.keys()),
                values=[count / total_count *
                        100 for count in class_counts.values()],
                title="% of each class in the audio file",
                title_font=dict(size=20),  # Adjust the font size
            )
        ]
        pie_chart = go.Figure(data=pie_chart_data)

        # Return the bar chart and pie chart, hide the loading spinner
        return [
            html.Div(
                f"The number of vehicles in the audio is: {total_count}"),
            html.Div(
                id="class-counts-graph",
                children=[
                    dcc.Graph(figure=bar_chart)
                ]
            ),
            html.Div(
                id="class-percentages-graph",
                children=[
                    dcc.Graph(figure=pie_chart)
                ]
            )
        ], {'display': 'block'}, ""
    else:
        return no_update, {'display': 'none'}, ""


# Callback for Page 2


@app.callback(
    Output('output-prediction', 'children'),
    Output('loading-output-page2', 'style'),
    Output('loading-output-page2', 'children'),
    Input('upload-audio-page2', 'contents')
)
def update_prediction(audio_content):
    if audio_content is not None:
        content_type, content_string = audio_content.split(',')
        decoded = base64.b64decode(content_string)

        # Load the audio file and extract MFCC features
        audio, sample_rate = librosa.load(io.BytesIO(
            decoded), res_type='kaiser_fast', duration=5)
        mfccs_features = librosa.feature.mfcc(
            y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        pre = mfccs_scaled_features.reshape(1, 1, 40)
        prediction = model.predict(pre)
        classes_x = np.argmax(prediction, axis=1)
        prediction_class = labelencoder.inverse_transform(classes_x)

        # Get the path of the corresponding image
        image_path = f"Vehicule Images/{prediction_class[0]}.png"

        # Check if the image file exists
        if os.path.exists(image_path):
            # Read the image file and encode it as base64
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(
                    image_file.read()).decode("utf-8")

        image_element = html.Img(
            src=f"data:image/png;base64,{encoded_image}", width="300", height="200", className='img-fluid mt-3')

        # Return the predicted class, image, and audio player button
        return (
            html.Div([
                html.P(f"Predicted Class: {prediction_class[0]}", style={
                    "color": "#1976D2"}),
                html.Button('Play Audio', id='play-audio-button',
                            className='btn btn-secondary mt-3 d-block'),
                html.Div(id='audio-player'),
                image_element
            ]),
            {'display': 'block'}, ""
        )
    else:
        return no_update, {'display': 'none'}, ""


# Callback to handle audio playback
@app.callback(Output('audio-player', 'children'),
              Input('play-audio-button', 'n_clicks'),
              State('upload-audio-page2', 'contents'))
def play_audio(n_clicks, audio_content):
    if n_clicks and n_clicks > 0 and audio_content is not None:
        content_type, content_string = audio_content.split(',')
        decoded = base64.b64decode(content_string)

        # Create an Audio element to play the audio
        audio_element = {
            'src': 'data:audio/mpeg;base64,' + base64.b64encode(decoded).decode(),
            'controls': 'controls'}

        # Return the audio element
        return html.Audio(**audio_element)

    return None


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
