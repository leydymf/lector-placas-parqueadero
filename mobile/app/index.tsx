import { CameraView, useCameraPermissions } from "expo-camera";
import * as Speech from "expo-speech";
import React, { useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Button,
  Image,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

export default function CameraScreen() {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [ip, setIp] = useState("");
  const [port, setPort] = useState("8080");
  const [image, setImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [plates, setPlates] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const apiUrl = ip && port ? `http://${ip}:${port}` : "";

  // --- Tomar foto y enviar al backend ---
  const handleCapture = async () => {
    if (!cameraRef.current) return;

    if (!ip) {
      Alert.alert("Error", "Ingresa la direccion IP del servidor.");
      return;
    }

    try {
      setLoading(true);
      const photo = await cameraRef.current.takePictureAsync({ base64: true });
      if (!photo) return;

      setImage(photo.uri);
      setPlates([]);
      setProcessedImage(null);

      // Enviar imagen base64 al backend via POST /predict/
      const fullUrl = `${apiUrl}/predict/`;
      console.log("Enviando imagen a:", fullUrl);

      const formBody = new URLSearchParams();
      formBody.append("image_base64", photo.base64 ?? "");

      const response = await fetch(fullUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
          Accept: "application/json",
        },
        body: formBody.toString(),
      });

      if (!response.ok) {
        const text = await response.text();
        console.error("Error HTTP:", response.status, text);
        Alert.alert("Error HTTP", `Codigo: ${response.status}`);
        hablar("Ocurrio un error al contactar el servidor.");
        return;
      }

      const data = await response.json();
      console.log("Respuesta del servidor:", data);

      // Procesar respuesta
      if (data?.placas && data.placas.length > 0) {
        setPlates(data.placas);

        if (data.image) {
          setProcessedImage(`data:image/jpeg;base64,${data.image}`);
        }

        const texto =
          data.placas.length === 1
            ? `La placa detectada es ${data.placas[0].split("").join(" ")}`
            : `Se detectaron ${data.placas.length} placas: ${data.placas.join(", ")}`;
        hablar(texto);
      } else if (data?.placas?.length === 0) {
        hablar("No se detectaron placas.");
        Alert.alert("Resultado", "No se detectaron placas.");
        setPlates([]);
        setProcessedImage(null);
      } else if (data?.error) {
        Alert.alert("Error del servidor", data.error);
        hablar("Ocurrio un error en el servidor.");
      }
    } catch (error) {
      console.error("Error enviando imagen:", error);
      Alert.alert("Error", "No se pudo conectar al servidor.");
      hablar("No se pudo conectar al servidor.");
    } finally {
      setLoading(false);
    }
  };

  // --- Text-to-speech en espanol ---
  const hablar = (texto: string) => {
    if (Platform.OS !== "web") {
      Speech.speak(texto, { language: "es-ES" });
    }
  };

  // --- Pantallas de permisos ---
  if (!permission) {
    return (
      <View style={styles.container}>
        <Text>Solicitando permisos...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.label}>Se necesita permiso para usar la camara.</Text>
        <Button title="Conceder permiso" onPress={requestPermission} />
      </View>
    );
  }

  // --- Pantalla principal ---
  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Lector de Placas</Text>
      <Text style={styles.subtitle}>UNAB - Ciencia de Datos</Text>

      {/* Configuracion del servidor */}
      <Text style={styles.label}>IP del servidor (AWS EC2):</Text>
      <TextInput
        style={styles.input}
        placeholder="Ej: 54.123.45.67"
        value={ip}
        onChangeText={setIp}
        autoCapitalize="none"
        autoCorrect={false}
      />

      <Text style={styles.label}>Puerto:</Text>
      <TextInput
        style={styles.input}
        placeholder="8080"
        value={port}
        onChangeText={setPort}
        keyboardType="numeric"
      />

      {/* Camara */}
      <CameraView ref={cameraRef} style={styles.camera} facing="back" />

      {/* Boton de captura */}
      <View style={styles.buttonContainer}>
        <Button
          title={loading ? "Procesando..." : "Detectar Placa"}
          onPress={handleCapture}
          color="#007AFF"
          disabled={loading}
        />
      </View>

      {loading && (
        <ActivityIndicator size="large" color="#007AFF" style={{ marginTop: 16 }} />
      )}

      {/* Imagen capturada */}
      {image && (
        <View style={styles.imageContainer}>
          <Text style={styles.label}>Imagen capturada:</Text>
          <Image source={{ uri: image }} style={styles.image} />
        </View>
      )}

      {/* Imagen procesada por el servidor */}
      {processedImage && (
        <View style={styles.imageContainer}>
          <Text style={styles.label}>Imagen procesada por el servidor:</Text>
          <Image
            source={{ uri: processedImage }}
            style={styles.image}
            resizeMode="contain"
          />
        </View>
      )}

      {/* Placas detectadas */}
      {plates.length > 0 && (
        <View style={styles.resultContainer}>
          <Text style={styles.label}>Placas detectadas:</Text>
          {plates.map((p, i) => (
            <Text key={i} style={styles.plateText}>
              {p}
            </Text>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    alignItems: "center",
    justifyContent: "flex-start",
    backgroundColor: "#f5f5f5",
    padding: 16,
    paddingTop: 48,
  },
  title: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#007AFF",
    marginBottom: 2,
  },
  subtitle: {
    fontSize: 13,
    color: "#888",
    marginBottom: 16,
  },
  label: {
    fontWeight: "bold",
    marginBottom: 6,
    color: "#333",
    alignSelf: "flex-start",
    marginLeft: "5%",
  },
  input: {
    width: "90%",
    height: 40,
    borderColor: "#ccc",
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 10,
    marginBottom: 10,
    backgroundColor: "#fff",
  },
  camera: {
    width: "100%",
    height: 400,
    borderRadius: 12,
    overflow: "hidden",
    marginBottom: 16,
  },
  buttonContainer: {
    width: "90%",
    marginBottom: 8,
  },
  imageContainer: {
    marginTop: 16,
    alignItems: "center",
  },
  image: {
    width: 300,
    height: 200,
    borderRadius: 10,
  },
  resultContainer: {
    marginTop: 20,
    backgroundColor: "#007AFF20",
    padding: 12,
    borderRadius: 8,
    width: "90%",
    marginBottom: 32,
  },
  plateText: {
    fontSize: 28,
    fontWeight: "bold",
    color: "#007AFF",
    textAlign: "center",
    marginVertical: 4,
  },
});
