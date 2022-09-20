const path = require("path");

module.exports = {
  env: {
    browser: true,
    es2021: true,
  },
  parser: "@typescript-eslint/parser",
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
    ecmaFeatures: {
      jsx: true,
    },
    tsconfigRootDir: __dirname,
  },

  plugins: ["react"],

  extends: [
    "prettier",
    "plugin:react/recommended",
    "standard-with-typescript",
    "plugin:i18next/recommended",
    "plugin:i18n-json/recommended",
  ],
  settings: {
    react: {
      version: "detect",
    },
  },
  rules: {
    // general things turned off for now
    "no-debugger": "warn",
    "eol-last": "off",
    "comma-dangle": ["off", "always-multiline"],
    "no-void": ["off"],
    "array-callback-return": ["off"],
    "spaced-comment": ["off"],
    "padded-blocks": ["off"],
    "no-multiple-empty-lines": ["off", { max: 2, maxEOF: 1 }],
    quotes: ["off", "double"],
    semi: ["off", "always"],
    yoda: ["off"],
    eqeqeq: ["off"],
    "react/display-name": "warn",

    // TS THINGS WE DONT WANT
    "@typescript-eslint/explicit-function-return-type": "off",
    "@typescript-eslint/ban-ts-comment": "off",

    // these are things that fight with prettier
    "@typescript-eslint/comma-dangle": "off",
    "@typescript-eslint/space-before-function-paren": "off",
    "@typescript-eslint/quotes": "off",
    "@typescript-eslint/semi": "off",
    "@typescript-eslint/brace-style": "off",
    "@typescript-eslint/indent": "off",
    "@typescript-eslint/member-delimiter-style": "off",

    // TS WARNINGS WE WANT
    "@typescript-eslint/no-unused-vars": "warn",
    "@typescript-eslint/no-non-null-assertion": "warn",

    // i18n stuff no string literal works but turned off for now
    "i18next/no-literal-string": "off",
    // still need to figure out how to get this to work
    // it should error if we dont haev all the keys in the translation file
    "i18n-json/identical-keys": [
      "error",
      {
        filePath: {
          "home.json/": path.resolve("./Translation/locales/en/home.json"),
        },
      },
    ],
  },
  overrides: [
    {
      files: ["*.ts", "*.tsx"],
      parserOptions: {
        project: ["./tsconfig.json"], // Specify it only for TypeScript files
      },
    },
  ],
  // eslint-disable-next-line semi
};
