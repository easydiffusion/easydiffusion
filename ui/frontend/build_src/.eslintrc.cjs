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
  },
  plugins: ["react"],
  extends: [
    "plugin:react/recommended",
    "standard-with-typescript",
    "plugin:i18next/recommended",
    "plugin:i18n-json/recommended",
  ],
  rules: {
    // general things turned off for now
    "prefer-const": "off",
    "no-debugger": "warn",

    "comma-dangle": ["off", "always-multiline"],
    "no-void": ["off"],
    "array-callback-return": ["off"],

    "react/display-name": "warn",
    quotes: ["off", "double"],
    semi: ["off", "always"],
    "no-multiple-empty-lines": ["off", { max: 2, maxEOF: 1 }],
    yoda: ["off"],
    eqeqeq: ["off"],
    "spaced-comment": ["off"],

    // TS  things turned off for now
    "@typescript-eslint/ban-ts-comment": "off",
    "@typescript-eslint/no-array-constructor": "off",
    "@typescript-eslint/explicit-function-return-type": "off",
    "@typescript-eslint/no-unused-vars": "off",
    "@typescript-eslint/naming-convention": "off",
    "@typescript-eslint/restrict-template-expressions": "off",
    "@typescript-eslint/prefer-optional-chain": "off",
    "@typescript-eslint/no-non-null-assertion": "off",
    "@typescript-eslint/strict-boolean-expressions": "off",
    "@typescript-eslint/no-floating-promises": "off",
    "@typescript-eslint/consistent-type-assertions": "off",
    "@typescript-eslint/comma-dangle": "off",
    "@typescript-eslint/quotes": "off",
    "@typescript-eslint/semi": "off",
    "@typescript-eslint/restrict-plus-operands": "off",
    "@typescript-eslint/brace-style": "off",
    "@typescript-eslint/prefer-ts-expect-error": "off",
    "@typescript-eslint/indent": "off",
    "@typescript-eslint/member-delimiter-style": "off",
    "@typescript-eslint/prefer-includes": "off",
    "@typescript-eslint/consistent-type-definitions": "off",
    "@typescript-eslint/no-unnecessary-condition": "off",
    "@typescript-eslint/no-unnecessary-type-assertion": "off",
    "@typescript-eslint/space-before-function-paren": "off",

    // i18n stuff no string literal works
    "i18next/no-literal-string": "warn",
    // still need to figure out how to get this to work
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
