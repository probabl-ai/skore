import type { ProjectItemDto } from "@/dto";
import { deserializeProjectItemDto, type PresentableItem } from "@/models";
import { describe, expectTypeOf, it } from "vitest";

describe("Project store", () => {
  it("Deserialize dto.", async () => {
    const now = new Date();
    const dto = {
      name: "pojhbv",
      media_type: "text/markdown",
      value: "- eins\n - zwei\n - polizei\n",
      updated_at: now.toISOString(),
      created_at: now.toISOString(),
    } as ProjectItemDto;

    const deserialized = deserializeProjectItemDto(dto);
    expectTypeOf(deserialized).toMatchTypeOf<PresentableItem>();
  });
});
