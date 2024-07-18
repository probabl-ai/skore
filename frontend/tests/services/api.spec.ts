import { fetchAllManderUris, fetchMander } from "@/services/api";
import { describe, expect, it } from "vitest";

import { createFetchResponse, mockedFetch } from "../test.utils";

describe("api", () => {
  it("Can fetch the list of manders from the server.", async () => {
    const uris = [
      "probabl-ai/demo-usecase/training/0",
      "probabl-ai/test-mandr/0",
      "probabl-ai/test-mandr/1",
      "probabl-ai/test-mandr/2",
      "probabl-ai/test-mandr/3",
      "probabl-ai/test-mandr/4",
    ];

    mockedFetch.mockResolvedValue(createFetchResponse(uris));

    const r = await fetchAllManderUris();
    expect(r).toStrictEqual(uris);
  });

  it("Can fetch a mander from the server", async () => {
    const mander = {
      path: "probabl-ai/demo-usecase/training/0",
      views: {
        confusion_matrix: "",
      },
      logs: {
        watermark:
          "Last updated: 2024-06-20T10:02:41.849437+00:00<br><br>Python implementation: CPython<br>Python version       : 3.12.3<br>IPython version      : 8.25.0<br><br>Compiler    : GCC 12.2.0<br>OS          : Linux<br>Release     : 6.6.22-linuxkit<br>Machine     : aarch64<br>Processor   : <br>CPU cores   : 8<br>Architecture: 64bit<br>",
      },
      artifacts: {
        model: {
          path: ".datamander/probabl-ai/demo-usecase/training/0/.artifacts/model.joblib",
        },
      },
      info: {
        X_shape: "[100, 20]",
        cv_results: "",
        n_classes: "2",
        train_time: "0.46034789085388184",
        updated_at: "1718877761",
      },
    };

    mockedFetch.mockResolvedValue(createFetchResponse(mander));

    const r = await fetchMander("random");
    expect(r).toStrictEqual(mander);
  });
});
