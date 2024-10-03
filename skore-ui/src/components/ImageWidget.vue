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
const src = computed(() => `data:${props.mediaType};base64,${props.base64Src}`);
</script>

<template>
  <img :src="src" :alt="alt" />
</template>
