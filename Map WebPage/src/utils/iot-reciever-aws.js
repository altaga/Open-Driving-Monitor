import { Component } from 'react';

class IotReciever extends Component {
    constructor(props) {
        super(props);
        this.state = {
            connect: false
        }
        this.mqttClient = ""
        this.mqttClientReady = false
        this.mount = true
    }

    componentDidUpdate(prevProps, prevState) {
        if(this.props.publishData){
            if (prevProps.publishData.topic !== this.props.publishData.topic && this.props.publishData.topic !== "") {
                this.mqttClient.publish(this.props.publishData.topic, this.props.publishData.message);
                this.props.callbackPublish()
            }
        }
    }

    async componentDidMount() {
        var AWS = require('aws-sdk');
        var AWSIoTData = require('aws-iot-device-sdk');
        var AWSConfiguration = require('./aws-configuration.js');
        AWS.config.region = AWSConfiguration.region;
        AWS.config.credentials = new AWS.CognitoIdentityCredentials({ IdentityPoolId: AWSConfiguration.poolId });
        var cognitoIdentity = new AWS.CognitoIdentity();
        var messageHistory = '';
        var refresh = 0;
        var clientId = 'mqtt-explorer-' + (Math.floor((Math.random() * 100000) + 1));

        this.mqttClient = AWSIoTData.device({
            region: AWS.config.region,
            host: AWSConfiguration.host,
            clientId: clientId,
            protocol: 'wss',
            maximumReconnectTimeMs: 1000,
            accessKeyId: '',
            secretKey: '',
            sessionToken: ''
        });
        console.log("Connecting...")
        await new Promise((resolve) => {
            AWS.config.credentials.get((err, data) => {
                if (!err) {
                    var params = { IdentityId: AWS.config.credentials.identityId };
                    cognitoIdentity.getCredentialsForIdentity(params, (err, data) => {
                        if (!err) {
                            this.mqttClient.updateWebSocketCredentials(data.Credentials.AccessKeyId, data.Credentials.SecretKey, data.Credentials.SessionToken);
                            resolve("ok")
                        }
                        else {
                            console.log('error retrieving credentials: ');
                        }
                    });
                }
                else { console.log('error retrieving identity:'); }
            });
        })

        this.mqttClient.mqttClientConnectHandler = () => {
            console.log("Connected IoT")
            console.log(this.props.subscribeTopics)
            for (let i = 0; i < this.props.subscribeTopics.length; i++) {
                this.mqttClient.subscribe(this.props.subscribeTopics[i]);
            }
            messageHistory = '';
            this.mount && this.setState({
                connect: true
            })
        }

        this.mqttClient.mqttClientReconnectHandler = () => {
            console.log('reconnect : times : ' + refresh.toString());
            this.mount && this.setState({
                connect: false
            })
        };

        this.mqttClient.mqttClientMessageHandler = (topic, payload) => {
            for (let i = 0; i < this.props.subscribeTopics.length; i++) {
                if (topic === this.props.subscribeTopics[i]) {
                    messageHistory = payload.toString()
                    this.props.callbackSubscribe({topic, message:messageHistory})
                }
            }
            messageHistory = "";
        }

        this.mqttClient.updateSubscriptionTopic = function () {

        };

        this.mqttClient.clearHistory = () => {

        };

        this.mqttClient.updatePublishTopic = () => { };

        this.mqttClient.updatePublishData = () => {

        };

        this.mqttClient.on('connect', this.mqttClient.mqttClientConnectHandler);
        this.mqttClient.on('reconnect', this.mqttClient.mqttClientReconnectHandler);
        this.mqttClient.on('message', this.mqttClient.mqttClientMessageHandler);
        this.mqttClient.on('close', () => {
            console.log('close IoT');
        });
        this.mqttClient.on('offline', () => {
            console.log('offline');
        });
        this.mqttClient.on('error', (error) => {
            console.log('error', error);
        });
    }

    componentWillUnmount() {
        this.mount = false
        this.mqttClient.end();
    }

    render() {
        return (
            <></>
        );
    }
}


export default IotReciever;