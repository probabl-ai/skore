import type { Client } from "@finos/perspective";
import perspective from "@finos/perspective";

let _perspectiveWorker: Client | null = null;

export async function getPerspectiveWorker() {
  if (_perspectiveWorker === null) {
    _perspectiveWorker = await perspective.worker();
  }
  return _perspectiveWorker;
}
