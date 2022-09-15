import { style, globalStyle } from "@vanilla-extract/css";

export const AdvancedSettingsList = style({
  // font-size: 9pt;
  // margin-bottom: 5px;
  // padding-left: 10px;
  // list-style-type: none;

  fontSize: "9pt",
  marginBottom: "5px",
  paddingLeft: "10px",
  listStyleType: "none",
});

export const AdvancedSettingItem = style({
  paddingBottom: "5px",
});

export const MenuButton = style({
  display: "block",
  width: "100%",
  textAlign: "left",
  backgroundColor: "transparent",
  color: "#fff",
  border: "0 none",
  cursor: "pointer",
  padding: "0",
  marginBottom: "10px",
});

globalStyle(`${MenuButton}> h4`, {
  color: "#e7ba71",
  marginTop: "5px !important",
});
