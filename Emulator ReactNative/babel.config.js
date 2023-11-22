module.exports = {
  presets: ['module:metro-react-native-babel-preset'],
  plugins: [
    [
      'module:react-native-dotenv',
      {
        envName: 'RN_ENV',
        moduleName: '@env',
        path: '.env.production',
        safe: true,
        allowUndefined: false,
        verbose: false,
      },
    ],
    //['@babel/plugin-transform-flow-strip-types'],
    //['@babel/plugin-transform-private-methods', {loose: true}],
  ],
};
