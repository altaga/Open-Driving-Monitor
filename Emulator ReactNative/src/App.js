import React, {Component} from 'react';
import reactAutobind from 'react-autobind';
import {View, Text, Pressable} from 'react-native';
import {Picker} from 'react-native-form-component';
import IotReciever from './utils/iot-reciever-aws';
import GlobalStyles from './styles/styles';
import GetLocation from 'react-native-get-location';

function getRandomInt(min, max) {
  return Math.floor(Math.random() * max) + min;
}

const emotions = [
  'Angry',
  'Disgust',
  'Fear',
  'Happy',
  'Sad',
  'Surprise',
  'Neutral',
];

const emotionsColors = {
  Angry: '#FF0000',
  Disgust: '#800080',
  Fear: '#000080',
  Happy: '#FFD700',
  Sad: '#0000FF',
  Surprise: '#00FF00',
  Neutral: '#808080',
};

const stateLabels = ['Awake', 'Drowsiness'];

const baseState = {
  enableGPS: false,
  enableIoT: false,
  publish: {message: '', topic: ''},
  pressed: false,
  emotion: emotions.map(item => ({
    label: item,
    value: item,
  }))[6],
  state: stateLabels.map(item => ({
    label: item,
    value: item,
  }))[0],
  location: {latitude: 0.0, longitude: 0.0},
};

/**
  { "coordinates": [gps["lng"],gps["lat"]], "color":class_colors[emotion["emotion"]],  "data":"Emotion: {}\nState: {}".format(emotion["emotion"],drowsinessState), "id": id}
*/

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {...baseState};
    reactAutobind(this);
    this.randomId = getRandomInt(0, 100_000_000_000);
    this.interval = null;
    this.gpsInterval = null;
  }

  getLocation() {
    GetLocation.getCurrentPosition({
      enableHighAccuracy: true,
      timeout: 30000,
    })
      .then(location => {
        const {longitude, latitude} = location;
        this.setState({
          location: {longitude, latitude},
          enableGPS: true,
        });
      })
      .catch(error => {
        const {code, message} = error;
        console.warn(code, message);
        this.setState({
          enableGPS: false,
        });
      });
  }

  componentDidMount() {
    this.getLocation(); // Get Location
    this.gpsInterval = setInterval(() => {
      // Update Location
      this.getLocation();
    }, 30_000); // Refresh Rate
    this.interval = setInterval(() => {
      const {enableGPS, enableIoT, pressed, location, emotion, state} =
        this.state;
      if (enableGPS && enableIoT && pressed) {
        let message = {
          coordinates: [location.longitude, location.latitude],
          color: emotionsColors[emotion.value],
          data: `Emotion: ${emotion.value}\nState: ${state.label}`,
          id: this.randomId,
        };
        this.setState({
          publish: {message: JSON.stringify(message), topic: '/ODM/devices'},
        });
      }
    }, 5000);
  }

  componentWillUnmount() {
    this.interval && clearInterval(this.interval);
    this.gpsInterval && clearInterval(this.gpsInterval);
  }

  render() {
    return (
      <View style={GlobalStyles.container}>
        <View style={{position: 'absolute', top: 18, right: 18}}>
          <IotReciever
            callbackConnected={e =>
              this.setState({
                enableIoT: e,
              })
            }
            publishData={{...this.state.publish}}
            callbackPublish={() =>
              this.setState({publish: {message: '', topic: ''}})
            }
            subscribeTopics={['/ODM/devices']}
            callbackSubscribe={dataReceived => console.log(dataReceived)}
            headerColor={'#00FF00'}
          />
        </View>
        <View
          style={{
            height: '100%',
            width: '100%',
            justifyContent: 'center',
            alignItems: 'center',
          }}>
          <Text style={GlobalStyles.textInput}>Location: (Rounded)</Text>
          <View
            style={[GlobalStyles.pickerInputStyle, {borderColor: '#0000FF'}]}>
            <Text style={GlobalStyles.textInput}>{`Lng: ${epsilonRound(
              this.state.location.longitude,
              3,
            )}`}</Text>
            <View
              style={{
                backgroundColor: '#0000FF',
                height: 1,
                width: '100%',
                marginVertical: 5,
              }}
            />
            <Text style={GlobalStyles.textInput}>{`Lat: ${epsilonRound(
              this.state.location.latitude,
              3,
            )}`}</Text>
          </View>
          <Picker
            // Style Eq
            buttonStyle={[
              GlobalStyles.pickerInputStyle,
              {borderColor: '#FF0000'},
            ]}
            // Button Disappear
            iconWrapperStyle={GlobalStyles.iconWrapperStyle}
            // Uppper Label
            labelStyle={GlobalStyles.labelStyle}
            // Selected
            selectedValueStyle={GlobalStyles.selectedValueStyle}
            // Label
            label="Emotion:"
            // Selectors
            selectedValue={this.state.emotion.value}
            items={emotions.map((item, index) => ({
              label: item,
              value: item,
            }))}
            onSelection={emotion => {
              this.setState({
                emotion,
              });
            }}
            type="modal"
          />
          <Picker
            // Style Eq
            buttonStyle={[
              GlobalStyles.pickerInputStyle,
              {borderColor: '#00FF00'},
            ]}
            // Button Disappear
            iconWrapperStyle={GlobalStyles.iconWrapperStyle}
            // Uppper Label
            labelStyle={GlobalStyles.labelStyle}
            // Selected
            selectedValueStyle={GlobalStyles.selectedValueStyle}
            // Label
            label="State:"
            // Selectors
            selectedValue={this.state.state.value}
            items={stateLabels.map(item => ({
              label: item,
              value: item,
            }))}
            onSelection={state => {
              this.setState({
                state,
              });
            }}
            type="modal"
          />
          <View
            style={{
              backgroundColor: '#000000',
              height: 1,
              width: '90%',
              marginVertical: 20,
            }}
          />
          <Pressable
            style={[
              GlobalStyles.pickerInputStyle,
              {
                borderColor: '#000000',
                backgroundColor: this.state.pressed ? '#AAAAAA' : 'white',
              },
            ]}
            onPress={async () => {
              this.setState({
                pressed: !this.state.pressed,
              });
            }}>
            <Text style={[GlobalStyles.buttonTextStyle]}>
              {this.state.pressed ? 'Stop Emulation' : 'Start Emulation'}
            </Text>
          </Pressable>
        </View>
      </View>
    );
  }
}

export default App;
