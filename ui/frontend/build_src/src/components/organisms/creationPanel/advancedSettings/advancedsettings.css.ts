import { style, globalStyle } from '@vanilla-extract/css';

export const AdvancedSettingsList = style({
  // font-size: 9pt;
  // margin-bottom: 5px;
  // padding-left: 10px;
  // list-style-type: none;

  fontSize: '9pt',
  marginBottom: '5px',
  paddingLeft: '10px',
  listStyleType: 'none',

});

export const AdvancedSettingItem = style({
  paddingBottom: '5px',
});

globalStyle( 'button > h4', {
  color:'lightgrey' 
});