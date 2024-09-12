<script setup lang="tsx">
import { computed } from "vue";

import type { SupportedImageMimeType } from "@/models";

interface ImageWidgetProps {
  base64Src: string;
  mimeType: SupportedImageMimeType;
  alt?: string;
}

const props = withDefaults(defineProps<ImageWidgetProps>(), {
  alt: "",
});
const src = computed(() => `data:${props.mimeType};base64,${props.base64Src}`);
</script>

<template>
  <img :src="src" :alt="alt" />
</template>
