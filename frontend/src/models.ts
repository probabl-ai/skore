export interface DataStore {
  path: string;
  views: {
    [key: string]: any;
  };
  logs: {
    [key: string]: any;
  };
  artifacts: {
    [key: string]: any;
  };
  info: {
    [key: string]: any;
  };
}
