export interface DataStore {
  path: string;
  views: string[];
  logs: string[];
  artifacts: {
    [key: string]: any;
  };
  info: {
    [key: string]: any;
  };
}
