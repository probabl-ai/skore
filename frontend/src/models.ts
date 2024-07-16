export interface Mander {
  path: string;
  views: string[];
  logs: string[];
  artifacts: {
    [key: string]: {
      path: string;
    };
  };
  info: {
    [key: string]: any;
  };
}
