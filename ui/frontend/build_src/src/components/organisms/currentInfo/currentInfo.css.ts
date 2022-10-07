import { style } from '@vanilla-extract/css'

import {
  card as cardStyles,
} from '../../_recipes/card.css'

export const currentInfoMain = style([
  cardStyles(
    {
      backing: 'dark',
    }
  ),
  {
    // display: 'flex',
    // flexDirection: 'column',
    // justifyContent: 'center',
    // alignItems: 'center',
    // height: '100%',
    width: '250px',
    padding: '0 0 0 0',
  },
])
