<script setup lang="ts">
import { RouterView } from "vue-router";

import AppToolbar from "@/components/AppToolbar.vue";
import LoadingBars from "@/components/LoadingBars.vue";
import ModalDialog from "@/components/ModalDialog.vue";
import ToastNotificationArea from "@/components/ToastNotificationArea.vue";
</script>

<template>
  <div class="skore">
    <AppToolbar />
    <RouterView v-slot="{ Component }">
      <template v-if="Component">
        <Transition name="fade">
          <Suspense>
            <component :is="Component" />
            <template #fallback>
              <div class="loader">
                <LoadingBars />
              </div>
            </template>
          </Suspense>
        </Transition>
      </template>
    </RouterView>
  </div>
  <ToastNotificationArea />
  <ModalDialog />
</template>

<style scoped>
.skore {
  display: flex;
  flex-direction: row;

  .loader {
    display: flex;
    width: 100dvw;
    height: 100dvh;
    align-items: center;
    justify-content: center;
  }
}
</style>
