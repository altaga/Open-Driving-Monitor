"use client";
import { Feature } from "ol";
import Map from "ol/Map";
import View from "ol/View";
import { Polygon } from "ol/geom";
import TileLayer from "ol/layer/Tile";
import VectorLayer from "ol/layer/Vector";
import { fromLonLat } from "ol/proj";
import VectorSource from "ol/source/Vector";
import XYZ from "ol/source/XYZ";
import { Fill, Style, Stroke, Text } from "ol/style";
import { useEffect, useRef, useState } from "react";
import { carArray } from "../../utils/constants";
import IotReciever from "../../utils/iot-reciever-aws";

function hexToRGBA(hex) {
  var r = parseInt(hex.slice(1, 3), 16),
    g = parseInt(hex.slice(3, 5), 16),
    b = parseInt(hex.slice(5, 7), 16),
    a = parseInt(hex.slice(7, 9), 16) / 255;
  return [r, g, b, a];
}

function drawCar(center, scale) {
  const meters = (0.00001 / 1.11) * scale; // in Meters
  const coordinates = carArray.map((operation) => [
    center[0] + operation[0] * meters,
    center[1] + operation[1] * meters,
  ]);
  const transformed = coordinates.map((coord) => fromLonLat(coord));
  return transformed;
}

function createCar(coordinates, color, scale, text = "") {
  const car = new Polygon([drawCar(coordinates, scale)]);
  const circleFeature = new Feature({
    geometry: car,
  });
  const vectorSource = new VectorSource({
    projection: "EPSG:4326",
  });
  vectorSource.addFeatures([circleFeature]);
  const vectorLayer = new VectorLayer({
    source: vectorSource,
    style: [
      new Style({
        fill: new Fill({
          color: hexToRGBA(color + "77"),
        }),
        stroke: new Stroke({
          color: hexToRGBA(color + "FF"),
          width: 3,
        }),
        text: new Text({
          text,
          font: `bold 15px sans-serif`,
          offsetY: scale * 6,
          fill: new Fill({ color: hexToRGBA("#000000FF") }),
          overflow: true,
          textAlign: "center",
        }),
      }),
    ],
  });
  return vectorLayer;
}

export default function Home(params) {
  const map = useRef();
  const isMounted = useRef(false);
  const cars = useRef({});
  const [coords, setCoords] = useState([0, 0, 17]);

  const carModify = (vector, id) => {
    let temp = { ...cars.current };
    if (temp[id]) {
      map.current.removeLayer(temp[id]);
      temp[id] = vector;
      map.current.addLayer(temp[id]);
    } else {
      temp[id] = vector;
      map.current.addLayer(temp[id]);
    }
    cars.current = { ...temp };
  };

  const callbackSubscribe = (dataReceived) => {
    /*
    Input format:
      {
        "coordinates": [-99.1913, 19.3531],
        "color":"#00FFFF",
        "data":"Emotion: Neutral\nState: Drowsiness",
        "id":0
      }
    */
    try {
      const { message } = dataReceived;
      const messageJSON = JSON.parse(message);
      const { coordinates, color, data, id } = messageJSON;
      const car = createCar(coordinates, color, 10, data);
      carModify(car, id);
      console.log({ coordinates, color, data, id });
    } catch {
      console.log("Data Format Error:");
      console.log(messageJSON);
    }
  };

  useEffect(() => {
    if (isMounted.current) {
      console.log(coords);
      map.current = new Map({
        target: "map",
        layers: [
          new TileLayer({
            source: new XYZ({
              url: "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
              crossOrigin: "anonymous",
            }),
          }),
        ],
        view: new View({
          center: fromLonLat([coords[0], coords[1]]),
          zoom: coords[2],
        }),
      });
    }
  }, [coords]);

  useEffect(() => {
    isMounted.current = true;
    try {
      const temp = decodeURIComponent(params.params.location)
        .replace("@", "")
        .replace("z", "")
        .split(",")
        .map((e) => parseFloat(e));
      if (temp[0] && temp[1] && temp[2]) {
        setCoords([temp[1], temp[0], temp[2]]);
      } else {
        throw "Error";
      }
    } catch {
      navigator.geolocation.getCurrentPosition(
        (e) => {
          setCoords([e.coords.longitude, e.coords.latitude, 15]);
        },
        (e) => console.log(e),
        { enableHighAccuracy: true }
      );
    }
  }, []);

  return (
    <>
      <IotReciever
        // publishData={{ message: "", topic: "" }} // Not used
        // callbackPublish={() => this.setState({ publish: { message: "", topic: "" } }) } // Not used
        subscribeTopics={["/ODM/devices"]}
        callbackSubscribe={(dataReceived) => callbackSubscribe(dataReceived)}
      />
      <div id="map" style={{ height: "100vh", width: "100vw" }} />
    </>
  );
}