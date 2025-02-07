import type { ProjectItemDto } from "@/dto";
import { getErrorMessage } from "@/services/utils";

/**
 * A `PresentableItem` is a view model based on a ProjectItemDto.
 */
export interface PresentableItem {
  name: string;
  mediaType: string;
  data: any;
  createdAt: Date;
  updatedAt: Date;
  note?: string;
  version: number;
}

/**
 * Deserialize an object received from the backend.
 * @param dto a backend received dto
 * @returns a presentable data
 */
export function deserializeProjectItemDto(dto: ProjectItemDto): PresentableItem {
  const isBase64 = dto.media_type.endsWith(";base64");
  const isImage = dto.media_type.startsWith("image/");

  let data = dto.value;
  let mediaType = dto.media_type;
  try {
    if (isBase64 && !isImage) {
      data = atob(data);
    }
    if (typeof data == "string" && dto.media_type.includes("json")) {
      data = JSON.parse(data);
    }
  } catch (error) {
    data = `\`\`\`\n${getErrorMessage(error)}\n\`\`\``;
    mediaType = "text/markdown";
  }

  const createdAt = new Date(dto.created_at);
  const updatedAt = new Date(dto.updated_at);
  return {
    name: dto.name,
    mediaType,
    data,
    createdAt,
    updatedAt,
    note: dto.note,
    version: dto.version,
  };
}
