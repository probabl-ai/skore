<script setup lang="tsx">
import { computed } from "vue";

interface ImageWidgetProps {
  base64Src: string;
  mediaType: string;
  alt?: string;
}

const props = withDefaults(defineProps<ImageWidgetProps>(), {
  alt: "",
});
const src = computed(() => {
  const mediaType = props.mediaType.includes(";base64")
    ? props.mediaType
    : `${props.mediaType};base64`;
  return `data:${mediaType},${props.base64Src}`;
});
</script>

<template>
  <img :src="src" :alt="alt" />
</template>
