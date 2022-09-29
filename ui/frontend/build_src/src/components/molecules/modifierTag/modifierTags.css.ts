import { style, globalStyle } from "@vanilla-extract/css";

// .modifierTag.selected {
//   background-color: rgb(131, 11, 121);
// }


export const ModifierTagMain = style({
  display: "inline-block",
  padding: "6px",
  backgroundColor: "rgb(38, 77, 141)",
  color: "#fff",
  borderRadius: "5px",
  margin: "5px",
});

// export const ModifierTagSelected = style({
//   backgroundColor: "rgb(131, 11, 121)",
// });

globalStyle(`${ModifierTagMain}.selected`, {
  backgroundColor: "rgb(131, 11, 121)",
})

globalStyle(`${ModifierTagMain} p`, {
  margin: 0,
  textAlign: "center",
  marginBottom: "2px",
});


export const tagPreview = style({
  display: 'flex',
  justifyContent: 'center',
});

globalStyle(`${tagPreview} img`, {
  width: "90px",
  height: "100%",
  objectFit: "cover",
  objectPosition: "center",
});

