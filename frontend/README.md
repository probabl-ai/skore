# Dashboard

This sub directory aims at creating a single page application supporting Mandr.

It should build as a SPA and as a library of web components.

## Useful commands

| command             | action                                            |
|---------------------|---------------------------------------------------|
| `npm install`       | install depdendecies                              |
| `npm run dev`       | compile and hot-reload for development            |
| `npm run build`     | type-check, compile and minify for production     |
| `npm run test:unit` | run unit tests with [vitest](https://vitest.dev/) |
| `npm run lint`      | lint with [ESLint](https://eslint.org/)           |


## Recommended IDE Setup

[VSCode](https://code.visualstudio.com/) + [Volar](https://marketplace.visualstudio.com/items?itemName=Vue.volar) (and disable Vetur).

## Type Support for `.vue` Imports in TS

TypeScript cannot handle type information for `.vue` imports by default, so we replace the `tsc` CLI with `vue-tsc` for type checking. In editors, we need [Volar](https://marketplace.visualstudio.com/items?itemName=Vue.volar) to make the TypeScript language service aware of `.vue` types.
