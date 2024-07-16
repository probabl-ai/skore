<script setup lang="ts">
import { RouterView } from "vue-router";
import LoadingBars from "./components/LoadingBars.vue";
</script>

<template>
  <RouterView v-slot="{ Component }">
    <template v-if="Component">
      <Suspense>
        <component :is="Component" />
        <template #fallback>
          <LoadingBars />
        </template>
      </Suspense>
    </template>
  </RouterView>
</template>
