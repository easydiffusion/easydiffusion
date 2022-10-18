import { style, globalStyle } from '@vanilla-extract/css'
import { vars } from '../../../../../../styles/theme/index.css'
import { card } from '../../../../../_recipes/card.css';

export const matrixListMain = style([card(
  {
    level: 'down'
  }
), {

  maxHeight: '500px',
  overflow: 'auto',

  ':empty': {
    display: 'none'
  }
}]);

export const matrixListItem = style([card(
  {
    level: 1,
    rounded: false,
    backing: 'dark',
  }
), {

}]);

globalStyle(`${matrixListMain} > ${matrixListItem}`, {
  marginTop: vars.spacing.small,
});


export const matrixListPrompt = style({});

export const matrixListTags = style({
  background: vars.backgroundAccentMain,
  display: 'flex',
  flexWrap: 'wrap',
  gap: '0.5rem',

  ':empty': {
    display: 'none'
  }

});