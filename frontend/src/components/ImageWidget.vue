<script setup lang="tsx">
import { computed } from "vue";

interface ImageWidgetProps {
  base64Src: string;
  mimeType: "image/svg+xml" | "image/png" | "image/jpeg" | "image/webp";
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
