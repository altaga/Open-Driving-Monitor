import {Dimensions, Platform, StatusBar, StyleSheet} from 'react-native';

const screenHeight = Dimensions.get('screen').height;
const windowHeight = Dimensions.get('window').height;

export const StatusBarHeight = StatusBar.currentHeight;
export const NavigatorBarHeight = screenHeight - windowHeight;

const GlobalStyles = StyleSheet.create({
  // Globals Layout
  container: {
    // Global Style
    width: Dimensions.get('window').width,
    height: Dimensions.get('window').height,
    display: 'flex',
    justifyContent: 'space-around',
    alignItems: 'center',
    backgroundColor: 'white',
  },
  // Pressables
  buttonStyle: {
    width: Dimensions.get('window').width * 0.9,
    alignItems: 'center',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderTopRightRadius: 20,
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 5,
    borderTopLeftRadius: 5,
    backgroundColor: '#000000aa',
    padding: 8,
  },
  buttonTextStyle: {
    color: 'black',
    fontSize: 22,
    fontWeight: 'bold',
  },
  // Text Input
  textInput: {
    fontSize: 24,
    color: 'black',
    fontWeight: 'bold',
    marginLeft: '2.5%',
  },
  //// Picker
  pickerInputStyle: {
    width: Dimensions.get('window').width * 0.9,
    alignItems: 'center',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderTopRightRadius: 20,
    borderBottomLeftRadius: 20,
    borderBottomRightRadius: 5,
    borderTopLeftRadius: 5,
    backgroundColor: 'white',
    padding: 8,
    alignSelf: 'center',
    height: 'auto',
    marginVertical: 20,
  },
  iconWrapperStyle: {
    backgroundColor: '#00000000',
    width: 0,
  },
  labelStyle: {
    color: 'black',
    fontSize: 24,
    textAlign: 'center',
    marginLeft: '2.5%',
  },
  selectedValueStyle: {
    color: 'black',
    fontWeight:"bold",
    width: '80%',
    marginLeft: '10%',
    textAlign: 'center',
    fontSize: 24,
  },
});

export default GlobalStyles;
