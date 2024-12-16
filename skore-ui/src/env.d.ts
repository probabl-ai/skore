/// <reference types="vite/client" />

declare module "markdown-it-sub";
declare module "markdown-it-sup";
declare module "markdown-it-highlightjs/core";

interface Window {
  skoreWidgetId?: string;
}

declare module "*?base64" {
  const value: string;
  export = value;
}
